/**
 * Virtual-pipes Shallow Water Equations solver.
 *
 * All cell-indexed state lives in storage buffers, not textures — that side-steps
 * the WebGPU restriction that `rg32float` storage textures can only be opened
 * with the `chromium-experimental-read-write-storage-texture` feature, and also
 * avoids the per-stage storage-texture cap (4 per stage on most adapters).
 *
 * Buffer layout (one buffer per logical field, row-major idx = j*width + i):
 *   bed      : 2 f32 per cell  (terrain, total)
 *   water    : 2 f32 per cell  (h, h_prev)
 *   fluxLR   : 2 f32 per cell  (left, right)
 *   fluxUD   : 2 f32 per cell  (down, up)
 *   velocity : 2 f32 per cell  (u, v)
 *   chunkId  : 1 u32 per cell
 *   chunkVel : 4 f32 per cell  (vx, vy, vz, speed)
 *   boundary : 1 u32 per cell
 *   prevBed  : 1 f32 per cell
 *   hDelta   : 1 i32 per cell  (atomic, fixed-point)
 *   forceAccum: 6 i32 per chunk (fx, fy, fz, tx, ty, tz; atomic, fixed-point)
 */
import type { GPUContext } from './GPUContext.js';
import type { SimulationGrid } from './SimulationGrid.js';
import { ShaderSource } from './shaders/index.js';

export interface SolverOptions {
  readonly dt: number;
  readonly substepsPerFrame: number;
  readonly gravity?: number;
  readonly damping?: number;
  readonly maxChunks?: number;
}

const DEFAULT_OPTS: Required<Omit<SolverOptions, 'dt' | 'substepsPerFrame'>> = {
  gravity: 9.81,
  damping: 0.98,
  maxChunks: 256,
};

const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class VirtualPipesSolver {
  readonly ctx: GPUContext;
  readonly grid: SimulationGrid;
  readonly opts: SolverOptions & typeof DEFAULT_OPTS;

  readonly bed: GPUBuffer;
  readonly water: GPUBuffer;
  readonly fluxLR: GPUBuffer;
  readonly fluxUD: GPUBuffer;
  readonly velocity: GPUBuffer;
  readonly chunkId: GPUBuffer;
  readonly chunkVel: GPUBuffer;
  readonly boundary: GPUBuffer;
  readonly prevBed: GPUBuffer;
  readonly hDelta: GPUBuffer;
  readonly forceAccum: GPUBuffer;
  readonly chunkCOMs: GPUBuffer;

  readonly params: GPUBuffer;
  private paramsCpu: ArrayBuffer;
  private paramsView: DataView;

  private pipelines: Record<string, GPUComputePipeline> = {};

  constructor(ctx: GPUContext, grid: SimulationGrid, opts: SolverOptions) {
    this.ctx = ctx;
    this.grid = grid;
    this.opts = { ...DEFAULT_OPTS, ...opts };
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
    this.prevBed   = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'isenflow.prevBed' });
    this.hDelta    = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'isenflow.hDelta' });

    const chunkBytes = this.opts.maxChunks * 6 * 4;
    this.forceAccum = dev.createBuffer({ size: chunkBytes, usage: STORAGE_USAGE, label: 'isenflow.forceAccum' });
    this.chunkCOMs  = dev.createBuffer({
      size: this.opts.maxChunks * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'isenflow.chunkCOMs',
    });

    this.paramsCpu = new ArrayBuffer(32);
    this.paramsView = new DataView(this.paramsCpu);
    this.params = dev.createBuffer({
      size: this.paramsCpu.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'isenflow.params',
    });
    this.uploadParams();
    this.seedInitial();
    this.buildPipelines();
  }

  private uploadParams(): void {
    const dv = this.paramsView;
    dv.setUint32(0, this.grid.width, true);
    dv.setUint32(4, this.grid.height, true);
    dv.setFloat32(8, this.grid.dx, true);
    dv.setFloat32(12, this.opts.dt, true);
    dv.setFloat32(16, this.opts.gravity, true);
    dv.setFloat32(20, this.opts.damping, true);
    dv.setFloat32(24, this.grid.dx * this.grid.dx, true);
    dv.setFloat32(28, this.grid.dx, true);
    this.ctx.queue.writeBuffer(this.params, 0, this.paramsCpu);
  }

  private seedInitial(): void {
    const cells = this.grid.cells;
    // Bed: terrain channel from seed, total = terrain
    const bedData = new Float32Array(cells * 2);
    for (let i = 0; i < cells; i++) {
      const b = this.grid.bedSeed[i] ?? 0;
      bedData[i * 2] = b;
      bedData[i * 2 + 1] = b;
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
    // prevBed = current total bed
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

  step(): void {
    const encoder = this.ctx.device.createCommandEncoder({ label: 'isenflow.step' });
    const dx = Math.ceil(this.grid.width / 8);
    const dy = Math.ceil(this.grid.height / 8);

    for (let s = 0; s < this.opts.substepsPerFrame; s++) {
      const passF = encoder.beginComputePass({ label: 'fluxes' });
      passF.setPipeline(this.pipelines.fluxes!);
      passF.setBindGroup(0, this.bindFluxes(this.pipelines.fluxes!));
      passF.dispatchWorkgroups(dx, dy, 1);
      passF.end();

      const passU = encoder.beginComputePass({ label: 'update' });
      passU.setPipeline(this.pipelines.update!);
      passU.setBindGroup(0, this.bindUpdate(this.pipelines.update!));
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
    passD.setBindGroup(0, this.bindDisplace(this.pipelines.displace!));
    passD.dispatchWorkgroups(dx, dy, 1);
    passD.end();

    const passF = encoder.beginComputePass({ label: 'foldDelta' });
    passF.setPipeline(this.pipelines.foldDelta!);
    passF.setBindGroup(0, this.bindFoldDelta(this.pipelines.foldDelta!));
    passF.dispatchWorkgroups(dx, dy, 1);
    passF.end();

    const passS = encoder.beginComputePass({ label: 'snapshot' });
    passS.setPipeline(this.pipelines.snapshot!);
    passS.setBindGroup(0, this.bindSnapshot(this.pipelines.snapshot!));
    passS.dispatchWorkgroups(dx, dy, 1);
    passS.end();
    this.ctx.queue.submit([encoder.finish()]);
  }

  private bindFluxes(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.bed } },
        { binding: 2, resource: { buffer: this.water } },
        { binding: 3, resource: { buffer: this.fluxLR } },
        { binding: 4, resource: { buffer: this.fluxUD } },
        { binding: 5, resource: { buffer: this.boundary } },
      ],
    });
  }

  private bindUpdate(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.water } },
        { binding: 2, resource: { buffer: this.fluxLR } },
        { binding: 3, resource: { buffer: this.fluxUD } },
        { binding: 4, resource: { buffer: this.velocity } },
        { binding: 5, resource: { buffer: this.boundary } },
      ],
    });
  }

  private bindDisplace(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.bed } },
        { binding: 2, resource: { buffer: this.water } },
        { binding: 3, resource: { buffer: this.prevBed } },
        { binding: 4, resource: { buffer: this.hDelta } },
      ],
    });
  }

  private bindFoldDelta(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.water } },
        { binding: 2, resource: { buffer: this.hDelta } },
      ],
    });
  }

  private bindSnapshot(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.bed } },
        { binding: 2, resource: { buffer: this.prevBed } },
      ],
    });
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
    // Use a per-row writeBuffer for safety (avoids a full grid-sized round-trip).
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
      this.ctx.queue.writeBuffer(this.bed, rowOffsetCells * 2 * 4, packed.buffer, packed.byteOffset, rowBytes);
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
      this.chunkId, this.chunkVel, this.boundary, this.prevBed, this.hDelta,
      this.forceAccum, this.chunkCOMs, this.params,
    ]) {
      b.destroy();
    }
  }
}
