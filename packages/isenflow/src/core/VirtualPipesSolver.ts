/**
 * Virtual-pipes Shallow Water Equations solver.
 *
 * All cell-indexed state lives in storage buffers, not textures — that side-steps
 * the WebGPU restriction that `rg32float` storage textures can only be opened
 * with the `chromium-experimental-read-write-storage-texture` feature, and also
 * avoids the per-stage storage-texture cap (4 per stage on most adapters).
 *
 * Buffer layout (one buffer per logical field, row-major idx = j*width + i):
 *   bed            : 2 f32 per cell  (terrain, total)
 *   water          : 2 f32 per cell  (h, h_prev)
 *   fluxLR         : 2 f32 per cell  (left, right)
 *   fluxUD         : 2 f32 per cell  (down, up)
 *   velocity       : 2 f32 per cell  (u, v)
 *   chunkId        : 1 u32 per cell
 *   chunkVel       : 4 f32 per cell  (vx, vy, vz, speed)
 *   boundary       : 1 u32 per cell
 *   boundaryTargetH: 1 f32 per cell  (target depth for Inflow/Sea)
 *   prevBed        : 1 f32 per cell
 *   hDelta         : 1 i32 per cell  (atomic, fixed-point)
 *   forceAccum     : 6 i32 per chunk (fx, fy, fz, tx, ty, tz; atomic, fixed-point)
 */
import type { GPUContext } from './GPUContext.js';
import type { SimulationGrid } from './SimulationGrid.js';
import { ShaderSource } from './shaders/index.js';

export interface SolverOptions {
  readonly dt: number;
  readonly substepsPerFrame: number;
  readonly gravity?: number;
  /**
   * Fraction of flux retained per second (Dagenais 2018 ω, default 0.5).
   * Uploaded to the GPU as `pow(damping, dt)` so decay is dt-independent.
   */
  readonly damping?: number;
  readonly maxChunks?: number;
  /** Effective cross-section of a virtual pipe; default `dx²`. */
  readonly pipeArea?: number;
  /** Effective length of a virtual pipe; default `dx`. */
  readonly pipeLen?: number;
  /** Manning roughness coefficient (default 0). Set to 0.03 for natural channels. */
  readonly manningN?: number;
}

const DEFAULT_OPTS: Required<Omit<SolverOptions, 'dt' | 'substepsPerFrame' | 'pipeArea' | 'pipeLen'>> = {
  gravity: 9.81,
  damping: 0.5,
  maxChunks: 256,
  manningN: 0,
};

const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class VirtualPipesSolver {
  readonly ctx: GPUContext;
  readonly grid: SimulationGrid;
  readonly opts: SolverOptions & typeof DEFAULT_OPTS & { pipeArea: number; pipeLen: number };

  readonly bed: GPUBuffer;
  readonly water: GPUBuffer;
  readonly fluxLR: GPUBuffer;
  readonly fluxUD: GPUBuffer;
  readonly velocity: GPUBuffer;
  readonly chunkId: GPUBuffer;
  readonly chunkVel: GPUBuffer;
  readonly boundary: GPUBuffer;
  readonly boundaryTargetH: GPUBuffer;
  readonly prevBed: GPUBuffer;
  readonly hDelta: GPUBuffer;
  readonly forceAccum: GPUBuffer;
  readonly chunkCOMs: GPUBuffer;

  readonly params: GPUBuffer;
  private paramsCpu: ArrayBuffer;
  private paramsView: DataView;

  /**
   * CPU mirror of the *terrain* (.x) channel. Kept in sync by every write
   * helper so dynamic-body rasterizers can restore terrain when a body
   * vacates a cell without reading back from the GPU.
   */
  readonly terrainMirror: Float32Array;

  private pipelines: Record<string, GPUComputePipeline> = {};

  // Cached bind groups (P2 perf fix — never re-created per frame)
  private cachedBindGroups: Record<string, GPUBindGroup> = {};

  constructor(ctx: GPUContext, grid: SimulationGrid, opts: SolverOptions) {
    this.ctx = ctx;
    this.grid = grid;
    this.opts = {
      ...DEFAULT_OPTS,
      pipeArea: opts.pipeArea ?? grid.dx * grid.dx,
      pipeLen: opts.pipeLen ?? grid.dx,
      ...opts,
    };
    const dev = ctx.device;
    const cells = grid.width * grid.height;

    this.bed       = dev.createBuffer({ size: cells * 2 * 4, usage: STORAGE_USAGE, label: 'isenflow.bed' });
    this.water     = dev.createBuffer({ size: cells * 2 * 4, usage: STORAGE_USAGE, label: 'isenflow.water' });
    this.fluxLR    = dev.createBuffer({ size: cells * 2 * 4, usage: STORAGE_USAGE, label: 'isenflow.fluxLR' });
    this.fluxUD    = dev.createBuffer({ size: cells * 2 * 4, usage: STORAGE_USAGE, label: 'isenflow.fluxUD' });
    this.velocity  = dev.createBuffer({ size: cells * 2 * 4, usage: STORAGE_USAGE, label: 'isenflow.velocity' });
    this.chunkId   = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'isenflow.chunkId' });
    this.chunkVel  = dev.createBuffer({ size: cells * 4 * 4, usage: STORAGE_USAGE, label: 'isenflow.chunkVel' });
    this.boundary  = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'isenflow.boundary' });
    this.boundaryTargetH = dev.createBuffer({ size: cells * 4, usage: STORAGE_USAGE, label: 'isenflow.boundaryTargetH' });
    this.prevBed   = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'isenflow.prevBed' });
    this.hDelta    = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'isenflow.hDelta' });

    const chunkBytes = this.opts.maxChunks * 6 * 4;
    this.forceAccum = dev.createBuffer({ size: chunkBytes, usage: STORAGE_USAGE, label: 'isenflow.forceAccum' });
    this.chunkCOMs  = dev.createBuffer({
      size: Math.max(256, this.opts.maxChunks) * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'isenflow.chunkCOMs',
    });

    this.terrainMirror = new Float32Array(cells);

    // SimParams: 8 original fields (32 bytes) + manningN + originX + originZ + pad = 48 bytes
    this.paramsCpu = new ArrayBuffer(48);
    this.paramsView = new DataView(this.paramsCpu);
    this.params = dev.createBuffer({
      size: this.paramsCpu.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'isenflow.params',
    });
    this.uploadParams();
    this.seedInitial();
    this.buildPipelines();
    this.buildBindGroups();
  }

  /** Bytes-per-chunk in `forceAccum` (6 × i32). Useful for force readback ring sizing. */
  get forceAccumStrideBytes(): number {
    return 6 * 4;
  }

  /** Total size in bytes of the forceAccum buffer (maxChunks * 6 * 4). */
  get forceAccumBytes(): number {
    return this.opts.maxChunks * this.forceAccumStrideBytes;
  }

  get maxChunks(): number {
    return this.opts.maxChunks;
  }

  /** Enable/disable CPU-side buoyancy mode (disables GPU buoyancy kernel). */
  setCpuBuoyancyMode(enabled: boolean): void {
    this.paramsView.setFloat32(44, enabled ? 1.0 : 0.0, true);
    this.ctx.queue.writeBuffer(this.params, 0, this.paramsCpu);
  }

  private uploadParams(): void {
    const dv = this.paramsView;
    dv.setUint32(0, this.grid.width, true);
    dv.setUint32(4, this.grid.height, true);
    dv.setFloat32(8, this.grid.dx, true);
    dv.setFloat32(12, this.opts.dt, true);
    dv.setFloat32(16, this.opts.gravity, true);
    // Dagenais 2018 eq. (1): upload ζ = pow(ω, dt) for dt-independent damping
    dv.setFloat32(20, Math.pow(this.opts.damping, this.opts.dt), true);
    dv.setFloat32(24, this.opts.pipeArea, true);
    dv.setFloat32(28, this.opts.pipeLen, true);
    // Extended fields
    dv.setFloat32(32, this.opts.manningN, true);
    dv.setFloat32(36, this.grid.origin[0], true);  // originX
    dv.setFloat32(40, this.grid.origin[1], true);  // originZ
    // 44: cpuBuoyancyMode (default 0 = GPU buoyancy; 1 = CPU buoyancy)
    dv.setFloat32(44, 0, true);
    this.ctx.queue.writeBuffer(this.params, 0, this.paramsCpu);
  }

  private seedInitial(): void {
    const cells = this.grid.cells;
    const bedData = new Float32Array(cells * 2);
    for (let i = 0; i < cells; i++) {
      const b = this.grid.bedSeed[i] ?? 0;
      bedData[i * 2] = b;
      bedData[i * 2 + 1] = b;
      this.terrainMirror[i] = b;
    }
    this.ctx.queue.writeBuffer(this.bed, 0, bedData);

    if (this.grid.initialDepth > 0) {
      const water = new Float32Array(cells * 2);
      for (let i = 0; i < cells; i++) {
        water[i * 2] = this.grid.initialDepth;
        water[i * 2 + 1] = this.grid.initialDepth;
      }
      this.ctx.queue.writeBuffer(this.water, 0, water);
    }
    const prev = new Float32Array(cells);
    for (let i = 0; i < cells; i++) prev[i] = bedData[i * 2 + 1]!;
    this.ctx.queue.writeBuffer(this.prevBed, 0, prev);
  }

  private buildPipelines(): void {
    const dev = this.ctx.device;
    const make = (label: string, code: string) =>
      dev.createComputePipeline({
        label,
        layout: 'auto',
        compute: { module: dev.createShaderModule({ code, label }), entryPoint: 'main' },
      });
    this.pipelines.fluxes = make('fluxes', ShaderSource.computeFluxes);
    this.pipelines.update = make('update', ShaderSource.updateWater);
    this.pipelines.displace = make('displace', ShaderSource.applyDisplacement);
    this.pipelines.foldDelta = make('foldDelta', ShaderSource.foldHDelta);
    this.pipelines.snapshot = make('snapshot', ShaderSource.snapshotBed);
    this.pipelines.accumulate = make('accumulate', ShaderSource.accumulateForces);
    this.pipelines.zeroForces = make('zeroForces', ShaderSource.zeroForces);
  }

  /** P2 perf fix: cache bind groups at construction, reuse every frame. */
  private buildBindGroups(): void {
    const dev = this.ctx.device;
    const bg = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) =>
      dev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    this.cachedBindGroups.fluxes = bg(this.pipelines.fluxes!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.bed } },
      { binding: 2, resource: { buffer: this.water } },
      { binding: 3, resource: { buffer: this.fluxLR } },
      { binding: 4, resource: { buffer: this.fluxUD } },
      { binding: 5, resource: { buffer: this.boundary } },
    ]);

    this.cachedBindGroups.update = bg(this.pipelines.update!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.water } },
      { binding: 2, resource: { buffer: this.fluxLR } },
      { binding: 3, resource: { buffer: this.fluxUD } },
      { binding: 4, resource: { buffer: this.velocity } },
      { binding: 5, resource: { buffer: this.boundary } },
      { binding: 6, resource: { buffer: this.boundaryTargetH } },
    ]);

    this.cachedBindGroups.displace = bg(this.pipelines.displace!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.bed } },
      { binding: 2, resource: { buffer: this.water } },
      { binding: 3, resource: { buffer: this.prevBed } },
      { binding: 4, resource: { buffer: this.hDelta } },
    ]);

    this.cachedBindGroups.foldDelta = bg(this.pipelines.foldDelta!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.water } },
      { binding: 2, resource: { buffer: this.hDelta } },
    ]);

    this.cachedBindGroups.snapshot = bg(this.pipelines.snapshot!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.bed } },
      { binding: 2, resource: { buffer: this.prevBed } },
    ]);

    this.cachedBindGroups.accumulate = bg(this.pipelines.accumulate!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.bed } },
      { binding: 2, resource: { buffer: this.water } },
      { binding: 3, resource: { buffer: this.velocity } },
      { binding: 4, resource: { buffer: this.chunkId } },
      { binding: 5, resource: { buffer: this.chunkVel } },
      { binding: 6, resource: { buffer: this.forceAccum } },
      { binding: 7, resource: { buffer: this.chunkCOMs } },
    ]);

    this.cachedBindGroups.zeroForces = bg(this.pipelines.zeroForces!, [
      { binding: 0, resource: { buffer: this.forceAccum } },
    ]);
  }

  step(): void {
    const encoder = this.ctx.device.createCommandEncoder({ label: 'isenflow.step' });
    const dx = Math.ceil(this.grid.width / 8);
    const dy = Math.ceil(this.grid.height / 8);

    for (let s = 0; s < this.opts.substepsPerFrame; s++) {
      const passF = encoder.beginComputePass({ label: 'fluxes' });
      passF.setPipeline(this.pipelines.fluxes!);
      passF.setBindGroup(0, this.cachedBindGroups.fluxes!);
      passF.dispatchWorkgroups(dx, dy, 1);
      passF.end();

      const passU = encoder.beginComputePass({ label: 'update' });
      passU.setPipeline(this.pipelines.update!);
      passU.setBindGroup(0, this.cachedBindGroups.update!);
      passU.dispatchWorkgroups(dx, dy, 1);
      passU.end();
    }
    this.ctx.queue.submit([encoder.finish()]);
  }

  applyDisplacement(): void {
    const encoder = this.ctx.device.createCommandEncoder({ label: 'isenflow.displace' });
    const dx = Math.ceil(this.grid.width / 8);
    const dy = Math.ceil(this.grid.height / 8);

    const passD = encoder.beginComputePass({ label: 'displace' });
    passD.setPipeline(this.pipelines.displace!);
    passD.setBindGroup(0, this.cachedBindGroups.displace!);
    passD.dispatchWorkgroups(dx, dy, 1);
    passD.end();

    const passF = encoder.beginComputePass({ label: 'foldDelta' });
    passF.setPipeline(this.pipelines.foldDelta!);
    passF.setBindGroup(0, this.cachedBindGroups.foldDelta!);
    passF.dispatchWorkgroups(dx, dy, 1);
    passF.end();

    const passS = encoder.beginComputePass({ label: 'snapshot' });
    passS.setPipeline(this.pipelines.snapshot!);
    passS.setBindGroup(0, this.cachedBindGroups.snapshot!);
    passS.dispatchWorkgroups(dx, dy, 1);
    passS.end();
    this.ctx.queue.submit([encoder.finish()]);
  }

  /**
   * Zero the per-chunk force/torque accumulator. Call once per frame BEFORE
   * `accumulateForces`. Workgroup is 64-wide; we dispatch over the full
   * `maxChunks * 6` i32 element count.
   */
  zeroForces(): void {
    const elements = this.opts.maxChunks * 6;
    const groups = Math.ceil(elements / 64);
    const encoder = this.ctx.device.createCommandEncoder({ label: 'isenflow.zeroForces' });
    const pass = encoder.beginComputePass({ label: 'zeroForces' });
    pass.setPipeline(this.pipelines.zeroForces!);
    pass.setBindGroup(0, this.cachedBindGroups.zeroForces!);
    pass.dispatchWorkgroups(groups, 1, 1);
    pass.end();
    this.ctx.queue.submit([encoder.finish()]);
  }

  /**
   * Run the per-cell water→body force kernel, summing into `forceAccum`.
   * Optionally records the accumulator copy for a `ForceReadback` ring.
   */
  accumulateForces(opts?: {
    copyTo?: GPUBuffer;
    onEncoder?: (encoder: GPUCommandEncoder) => void;
  }): void {
    const dx = Math.ceil(this.grid.width / 8);
    const dy = Math.ceil(this.grid.height / 8);
    const encoder = this.ctx.device.createCommandEncoder({ label: 'isenflow.accumulate' });
    const pass = encoder.beginComputePass({ label: 'accumulate' });
    pass.setPipeline(this.pipelines.accumulate!);
    pass.setBindGroup(0, this.cachedBindGroups.accumulate!);
    pass.dispatchWorkgroups(dx, dy, 1);
    pass.end();

    if (opts?.copyTo) {
      encoder.copyBufferToBuffer(this.forceAccum, 0, opts.copyTo, 0, this.forceAccumBytes);
    }
    if (opts?.onEncoder) opts.onEncoder(encoder);

    this.ctx.queue.submit([encoder.finish()]);
  }

  // ---------- write helpers (called by demos / heightfield rasterizer) ----------

  /**
   * Set the full bed buffer from a CPU array of length width*height with
   * single-channel elevations. Both .x (terrain) and .y (total) are set.
   */
  writeBedFull(values: Float32Array): void {
    if (values.length !== this.grid.cells) {
      throw new Error(`writeBedFull: expected ${this.grid.cells} values, got ${values.length}`);
    }
    const packed = new Float32Array(this.grid.cells * 2);
    for (let i = 0; i < values.length; i++) {
      packed[i * 2] = values[i]!;
      packed[i * 2 + 1] = values[i]!;
      this.terrainMirror[i] = values[i]!;
    }
    this.ctx.queue.writeBuffer(this.bed, 0, packed);
  }

  /**
   * Stamp `values` (row-major, length region.w*region.h, single channel) into
   * the bed buffer. Both terrain and total channels are overwritten.
   */
  writeBedRegion(region: { x: number; y: number; w: number; h: number }, values: Float32Array): void {
    if (values.length !== region.w * region.h) {
      throw new Error('writeBedRegion: values length mismatch');
    }
    const W = this.grid.width;
    for (let row = 0; row < region.h; row++) {
      const rowOffsetCells = (region.y + row) * W + region.x;
      const rowBytes = region.w * 2 * 4;
      const packed = new Float32Array(region.w * 2);
      for (let i = 0; i < region.w; i++) {
        const v = values[row * region.w + i]!;
        packed[i * 2] = v;
        packed[i * 2 + 1] = v;
        this.terrainMirror[rowOffsetCells + i] = v;
      }
      this.ctx.queue.writeBuffer(this.bed, rowOffsetCells * 2 * 4, packed.buffer, packed.byteOffset, rowBytes);
    }
  }

  /**
   * Stamp the total-bed channel only, leaving terrain intact.
   * P4 perf fix: read back terrain from CPU mirror, pack full 2-float rows,
   * one writeBuffer per row instead of per cell.
   */
  writeBedTotalRegion(
    region: { x: number; y: number; w: number; h: number },
    values: Float32Array,
  ): void {
    if (values.length !== region.w * region.h) {
      throw new Error('writeBedTotalRegion: values length mismatch');
    }
    const W = this.grid.width;
    for (let row = 0; row < region.h; row++) {
      const rowOffsetCells = (region.y + row) * W + region.x;
      const packed = new Float32Array(region.w * 2);
      for (let i = 0; i < region.w; i++) {
        packed[i * 2] = this.terrainMirror[rowOffsetCells + i]!;
        packed[i * 2 + 1] = values[row * region.w + i]!;
      }
      this.ctx.queue.writeBuffer(this.bed, rowOffsetCells * 2 * 4, packed.buffer, packed.byteOffset, region.w * 2 * 4);
    }
  }

  writeWaterFull(values: Float32Array): void {
    if (values.length !== this.grid.cells) {
      throw new Error(`writeWaterFull: expected ${this.grid.cells} values, got ${values.length}`);
    }
    const packed = new Float32Array(this.grid.cells * 2);
    for (let i = 0; i < values.length; i++) {
      packed[i * 2] = values[i]!;
      packed[i * 2 + 1] = values[i]!;
    }
    this.ctx.queue.writeBuffer(this.water, 0, packed);
  }

  writeWaterRegion(region: { x: number; y: number; w: number; h: number }, values: Float32Array): void {
    if (values.length !== region.w * region.h) {
      throw new Error('writeWaterRegion: values length mismatch');
    }
    const W = this.grid.width;
    for (let row = 0; row < region.h; row++) {
      const rowOffsetCells = (region.y + row) * W + region.x;
      const rowBytes = region.w * 2 * 4;
      const packed = new Float32Array(region.w * 2);
      for (let i = 0; i < region.w; i++) {
        const v = values[row * region.w + i]!;
        packed[i * 2] = v;
        packed[i * 2 + 1] = v;
      }
      this.ctx.queue.writeBuffer(this.water, rowOffsetCells * 2 * 4, packed.buffer, packed.byteOffset, rowBytes);
    }
  }

  writeWaterCell(i: number, j: number, depth: number): void {
    const data = new Float32Array([depth, depth]);
    const offset = (j * this.grid.width + i) * 2 * 4;
    this.ctx.queue.writeBuffer(this.water, offset, data);
  }

  /** Stamp a region of the per-cell `chunkId` buffer. */
  writeChunkIdRegion(
    region: { x: number; y: number; w: number; h: number },
    chunkId: number,
  ): void {
    const W = this.grid.width;
    const row = new Uint32Array(region.w).fill(chunkId);
    for (let r = 0; r < region.h; r++) {
      const rowOffsetCells = (region.y + r) * W + region.x;
      this.ctx.queue.writeBuffer(this.chunkId, rowOffsetCells * 4, row.buffer, row.byteOffset, region.w * 4);
    }
  }

  /**
   * Stamp a region of the per-cell `chunkVel` buffer (4 floats per cell:
   * vx, vy, vz, speed).
   */
  writeChunkVelRegion(
    region: { x: number; y: number; w: number; h: number },
    vx: number,
    vy: number,
    vz: number,
  ): void {
    const W = this.grid.width;
    const speed = Math.hypot(vx, vy, vz);
    const row = new Float32Array(region.w * 4);
    for (let i = 0; i < region.w; i++) {
      row[i * 4 + 0] = vx;
      row[i * 4 + 1] = vy;
      row[i * 4 + 2] = vz;
      row[i * 4 + 3] = speed;
    }
    for (let r = 0; r < region.h; r++) {
      const rowOffsetCells = (region.y + r) * W + region.x;
      this.ctx.queue.writeBuffer(
        this.chunkVel,
        rowOffsetCells * 16,
        row.buffer,
        row.byteOffset,
        region.w * 16,
      );
    }
  }

  /**
   * Zero all dynamic simulation state (flux, velocity, chunks).
   * Called on demo switch to prevent stale momentum bleeding across demos.
   */
  resetSimState(): void {
    const cells = this.grid.cells;
    const zero2 = new Float32Array(cells * 2);
    this.ctx.queue.writeBuffer(this.fluxLR, 0, zero2);
    this.ctx.queue.writeBuffer(this.fluxUD, 0, zero2);
    this.ctx.queue.writeBuffer(this.velocity, 0, zero2);
    this.clearChunkIds();
    this.clearChunkVel();
    this.setCpuBuoyancyMode(false);
  }

  /** Reset the chunk-id buffer to zero (no chunks anywhere). */
  clearChunkIds(): void {
    const zeros = new Uint32Array(this.grid.cells);
    this.ctx.queue.writeBuffer(this.chunkId, 0, zeros);
  }

  /** Reset the chunk-velocity buffer. */
  clearChunkVel(): void {
    const zeros = new Float32Array(this.grid.cells * 4);
    this.ctx.queue.writeBuffer(this.chunkVel, 0, zeros);
  }

  /**
   * Upload per-chunk centre-of-mass + half-height-Y as a single (x, y, z, halfY)
   * vec4 array, sized for `maxChunks`. The shader uses `halfY` to derive
   * the body's top/bottom for buoyancy and vertical-damping forces.
   *
   * `data.length` must be `≥ maxChunks * 4`.
   */
  writeChunkCOMs(data: Float32Array): void {
    const required = this.opts.maxChunks * 4;
    const padded = data.length === required ? data : (() => {
      const p = new Float32Array(required);
      p.set(data.subarray(0, Math.min(data.length, required)));
      return p;
    })();
    this.ctx.queue.writeBuffer(this.chunkCOMs, 0, padded.buffer, padded.byteOffset, padded.byteLength);
  }

  /** Set a single chunk's COM (x, y, z) and half-height-Y. */
  writeChunkCOM(chunkId: number, x: number, y: number, z: number, halfY = 0): void {
    if (chunkId < 0 || chunkId >= this.opts.maxChunks) {
      throw new Error(`writeChunkCOM: chunkId ${chunkId} out of range [0, ${this.opts.maxChunks})`);
    }
    const arr = new Float32Array([x, y, z, halfY]);
    this.ctx.queue.writeBuffer(this.chunkCOMs, chunkId * 16, arr);
  }

  /** Stamp a region of the per-cell boundary buffer. */
  writeBoundaryRegion(
    region: { x: number; y: number; w: number; h: number },
    boundaryType: number,
  ): void {
    const W = this.grid.width;
    const row = new Uint32Array(region.w).fill(boundaryType);
    for (let r = 0; r < region.h; r++) {
      const rowOffsetCells = (region.y + r) * W + region.x;
      this.ctx.queue.writeBuffer(this.boundary, rowOffsetCells * 4, row.buffer, row.byteOffset, region.w * 4);
    }
  }

  /**
   * Set both boundary type and target depth for a region. Inflow (4) uses
   * the target as a floor; Sea (5) pins to it exactly.
   */
  writeBoundaryRegionTarget(
    region: { x: number; y: number; w: number; h: number },
    boundaryType: number,
    targetDepth: number,
  ): void {
    this.writeBoundaryRegion(region, boundaryType);
    const W = this.grid.width;
    const row = new Float32Array(region.w).fill(targetDepth);
    for (let r = 0; r < region.h; r++) {
      const rowOffsetCells = (region.y + r) * W + region.x;
      this.ctx.queue.writeBuffer(this.boundaryTargetH, rowOffsetCells * 4, row.buffer, row.byteOffset, region.w * 4);
    }
  }

  // ---------- readback helpers ----------

  private async readF32Buffer(buf: GPUBuffer): Promise<Float32Array> {
    const dst = this.ctx.device.createBuffer({
      size: buf.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.ctx.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, dst, 0, buf.size);
    this.ctx.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(dst.getMappedRange().slice(0));
    dst.unmap();
    dst.destroy();
    return out;
  }

  /** Returns length width*height*2 [h, h_prev] interleaved. */
  readWater(): Promise<Float32Array> { return this.readF32Buffer(this.water); }
  /** Returns length width*height*2 [terrain, total] interleaved. */
  readBed():   Promise<Float32Array> { return this.readF32Buffer(this.bed); }
  /** Returns length width*height*2 [u, v] interleaved. */
  readVelocity(): Promise<Float32Array> { return this.readF32Buffer(this.velocity); }

  /**
   * Read back the raw force accumulator buffer. Each chunk has 6 i32 entries
   * (fx, fy, fz, tx, ty, tz) in fixed-point — divide by `FIXED_POINT_SCALE`
   * (10000) to get Newtons / Newton-metres.
   */
  async readForceAccum(): Promise<Int32Array> {
    const dst = this.ctx.device.createBuffer({
      size: this.forceAccumBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.ctx.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.forceAccum, 0, dst, 0, this.forceAccumBytes);
    this.ctx.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const out = new Int32Array(dst.getMappedRange().slice(0));
    dst.unmap();
    dst.destroy();
    return out;
  }

  async totalVolume(): Promise<number> {
    const w = await this.readWater();
    let sum = 0;
    const area = this.grid.dx * this.grid.dx;
    for (let i = 0; i < w.length; i += 2) sum += Math.max(0, w[i]!) * area;
    return sum;
  }

  destroy(): void {
    for (const b of [
      this.bed, this.water, this.fluxLR, this.fluxUD, this.velocity,
      this.chunkId, this.chunkVel, this.boundary, this.boundaryTargetH,
      this.prevBed, this.hDelta,
      this.forceAccum, this.chunkCOMs, this.params,
    ]) {
      b.destroy();
    }
  }
}
