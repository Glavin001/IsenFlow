/**
 * Virtual-pipes Shallow Water Equations solver, WebGPU compute pipeline owner.
 *
 * Owns all storage textures and atomic buffers, exposes a per-frame `step()`
 * that runs N substeps, plus accessors for reading water/bed back to CPU.
 *
 * The compute kernels are in src/core/shaders/*.wgsl. This file wires them
 * together with bind-group layouts and dispatch sizes.
 */
import type { GPUContext } from './GPUContext.js';
import type { SimulationGrid } from './SimulationGrid.js';
import { ShaderSource } from './shaders/index.js';

export interface SolverOptions {
  readonly dt: number;             // simulation timestep (s)
  readonly substepsPerFrame: number;
  readonly gravity?: number;
  readonly damping?: number;       // pipe damping ∈ (0, 1]
  readonly maxChunks?: number;
}

const DEFAULT_OPTS: Required<Omit<SolverOptions, 'dt' | 'substepsPerFrame'>> = {
  gravity: 9.81,
  damping: 0.98,
  maxChunks: 256,
};

export class VirtualPipesSolver {
  readonly ctx: GPUContext;
  readonly grid: SimulationGrid;
  readonly opts: SolverOptions & typeof DEFAULT_OPTS;

  // Storage textures
  readonly bedTex: GPUTexture;
  readonly waterTex: GPUTexture;
  readonly fluxLR: GPUTexture;
  readonly fluxUD: GPUTexture;
  readonly velocity: GPUTexture;
  readonly chunkId: GPUTexture;
  readonly chunkVel: GPUTexture;
  readonly boundary: GPUTexture;
  readonly prevBed: GPUTexture;

  // Atomic storage buffers
  readonly forceAccum: GPUBuffer;
  readonly torqueAccum: GPUBuffer;
  readonly hDelta: GPUBuffer;
  readonly chunkCOMs: GPUBuffer;

  // Uniforms
  readonly params: GPUBuffer;
  private paramsCpu: ArrayBuffer;
  private paramsView: DataView;

  // Pipelines
  private pipelines: Record<string, GPUComputePipeline> = {};

  // Force readback ring
  private readonly readbackBuffers: GPUBuffer[] = [];
  private readonly readbackInFlight: boolean[] = [];

  constructor(ctx: GPUContext, grid: SimulationGrid, opts: SolverOptions) {
    this.ctx = ctx;
    this.grid = grid;
    this.opts = { ...DEFAULT_OPTS, ...opts };

    const w = grid.width;
    const h = grid.height;
    const dev = ctx.device;

    const usage =
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.TEXTURE_BINDING;

    this.bedTex = dev.createTexture({ size: [w, h], format: 'rg32float', usage, label: 'isenflow.bed' });
    this.waterTex = dev.createTexture({ size: [w, h], format: 'rg32float', usage, label: 'isenflow.water' });
    this.fluxLR = dev.createTexture({ size: [w, h], format: 'rg32float', usage, label: 'isenflow.fluxLR' });
    this.fluxUD = dev.createTexture({ size: [w, h], format: 'rg32float', usage, label: 'isenflow.fluxUD' });
    this.velocity = dev.createTexture({ size: [w, h], format: 'rg32float', usage, label: 'isenflow.velocity' });
    this.chunkId = dev.createTexture({ size: [w, h], format: 'r32uint', usage, label: 'isenflow.chunkId' });
    this.chunkVel = dev.createTexture({ size: [w, h], format: 'rgba32float', usage, label: 'isenflow.chunkVel' });
    this.boundary = dev.createTexture({ size: [w, h], format: 'r32uint', usage, label: 'isenflow.boundary' });
    this.prevBed = dev.createTexture({ size: [w, h], format: 'r32float', usage, label: 'isenflow.prevBed' });

    // Buffers
    const chunkBytes = this.opts.maxChunks * 3 * 4;
    this.forceAccum = dev.createBuffer({
      size: chunkBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'isenflow.forceAccum',
    });
    this.torqueAccum = dev.createBuffer({
      size: chunkBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'isenflow.torqueAccum',
    });
    this.hDelta = dev.createBuffer({
      size: w * h * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'isenflow.hDelta',
    });
    this.chunkCOMs = dev.createBuffer({
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

    // Readback ring (3 buffers).
    for (let i = 0; i < 3; i++) {
      this.readbackBuffers.push(
        dev.createBuffer({
          size: chunkBytes * 2, // force + torque
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          label: `isenflow.readback[${i}]`,
        }),
      );
      this.readbackInFlight.push(false);
    }

    this.seedBed();
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

  private seedBed(): void {
    const w = this.grid.width;
    const h = this.grid.height;
    const buf = new Float32Array(w * h * 2);
    for (let i = 0; i < w * h; i++) {
      const b = this.grid.bedSeed[i] ?? 0;
      buf[i * 2] = b;
      buf[i * 2 + 1] = b;
    }
    this.ctx.queue.writeTexture(
      { texture: this.bedTex },
      buf,
      { bytesPerRow: w * 8, rowsPerImage: h },
      { width: w, height: h, depthOrArrayLayers: 1 },
    );
    // initialize water
    if (this.grid.initialDepth > 0) {
      const wb = new Float32Array(w * h * 2);
      for (let i = 0; i < w * h; i++) {
        wb[i * 2] = this.grid.initialDepth;
        wb[i * 2 + 1] = this.grid.initialDepth;
      }
      this.ctx.queue.writeTexture(
        { texture: this.waterTex },
        wb,
        { bytesPerRow: w * 8, rowsPerImage: h },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
    }
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

  /** Run one render-frame step (N substeps internally). */
  step(): void {
    const encoder = this.ctx.device.createCommandEncoder({ label: 'isenflow.step' });
    const w = this.grid.width;
    const h = this.grid.height;
    const dx = Math.ceil(w / 8);
    const dy = Math.ceil(h / 8);
    const dispatchTiled = (pass: GPUComputePassEncoder) => pass.dispatchWorkgroups(dx, dy, 1);

    for (let s = 0; s < this.opts.substepsPerFrame; s++) {
      // 1. fluxes
      {
        const pipeline = this.pipelines.fluxes;
        const pass = encoder.beginComputePass({ label: 'fluxes' });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.bindFluxes(pipeline));
        dispatchTiled(pass);
        pass.end();
      }
      // 2. update water + velocity
      {
        const pipeline = this.pipelines.update;
        const pass = encoder.beginComputePass({ label: 'update' });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.bindUpdate(pipeline));
        dispatchTiled(pass);
        pass.end();
      }
    }

    this.ctx.queue.submit([encoder.finish()]);
  }

  /** Run the body→water displacement pass (call after rasterizing dynamic bodies). */
  applyDisplacement(): void {
    const encoder = this.ctx.device.createCommandEncoder({ label: 'isenflow.displace' });
    const w = this.grid.width;
    const h = this.grid.height;
    const dx = Math.ceil(w / 8);
    const dy = Math.ceil(h / 8);

    {
      const pipeline = this.pipelines.displace;
      const pass = encoder.beginComputePass({ label: 'displace' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindDisplace(pipeline));
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }
    {
      const pipeline = this.pipelines.foldDelta;
      const pass = encoder.beginComputePass({ label: 'foldDelta' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindFoldDelta(pipeline));
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }
    {
      const pipeline = this.pipelines.snapshot;
      const pass = encoder.beginComputePass({ label: 'snapshot' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindSnapshot(pipeline));
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }

    this.ctx.queue.submit([encoder.finish()]);
  }

  // --- bind helpers ------------------------------------------------------
  private bindFluxes(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: this.bedTex.createView() },
        { binding: 2, resource: this.waterTex.createView() },
        { binding: 3, resource: this.fluxLR.createView() },
        { binding: 4, resource: this.fluxUD.createView() },
        { binding: 5, resource: this.boundary.createView() },
      ],
    });
  }

  private bindUpdate(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: this.waterTex.createView() },
        { binding: 2, resource: this.fluxLR.createView() },
        { binding: 3, resource: this.fluxUD.createView() },
        { binding: 4, resource: this.velocity.createView() },
        { binding: 5, resource: this.boundary.createView() },
      ],
    });
  }

  private bindDisplace(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: this.bedTex.createView() },
        { binding: 2, resource: this.waterTex.createView() },
        { binding: 3, resource: this.prevBed.createView() },
        { binding: 4, resource: { buffer: this.hDelta } },
      ],
    });
  }

  private bindFoldDelta(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: this.waterTex.createView() },
        { binding: 2, resource: { buffer: this.hDelta } },
      ],
    });
  }

  private bindSnapshot(pipeline: GPUComputePipeline): GPUBindGroup {
    return this.ctx.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: this.bedTex.createView() },
        { binding: 2, resource: this.prevBed.createView() },
      ],
    });
  }

  // --- public helpers ----------------------------------------------------

  /**
   * Overwrite the dynamic bed (channel .y of bedTex) for the given region.
   * `region` is in cell coordinates. `values` length = region cells, in row-major.
   */
  writeBedRegion(region: { x: number; y: number; w: number; h: number }, values: Float32Array): void {
    if (values.length !== region.w * region.h) {
      throw new Error('writeBedRegion: values length mismatch');
    }
    const packed = new Float32Array(values.length * 2);
    for (let i = 0; i < values.length; i++) {
      // keep terrain (.x) unchanged via partial write; we will overwrite both channels.
      packed[i * 2] = values[i]!;
      packed[i * 2 + 1] = values[i]!;
    }
    this.ctx.queue.writeTexture(
      { texture: this.bedTex, origin: { x: region.x, y: region.y } },
      packed,
      { bytesPerRow: region.w * 8, rowsPerImage: region.h },
      { width: region.w, height: region.h, depthOrArrayLayers: 1 },
    );
  }

  /** Overwrite the water depth for a single cell. */
  writeWaterCell(i: number, j: number, depth: number): void {
    const data = new Float32Array([depth, depth]);
    this.ctx.queue.writeTexture(
      { texture: this.waterTex, origin: { x: i, y: j } },
      data,
      { bytesPerRow: 8, rowsPerImage: 1 },
      { width: 1, height: 1, depthOrArrayLayers: 1 },
    );
  }

  /**
   * Asynchronously read the current water-depth texture into a Float32Array.
   * Returns [r,g] interleaved (h, h_prev). Length = w * h * 2.
   */
  async readWater(): Promise<Float32Array> {
    return this.readRG32Texture(this.waterTex);
  }

  /** Same for bed (terrain, total). */
  async readBed(): Promise<Float32Array> {
    return this.readRG32Texture(this.bedTex);
  }

  /** Same for velocity (u, v). */
  async readVelocity(): Promise<Float32Array> {
    return this.readRG32Texture(this.velocity);
  }

  private async readRG32Texture(tex: GPUTexture): Promise<Float32Array> {
    const w = this.grid.width;
    const h = this.grid.height;
    // bytesPerRow must be a multiple of 256.
    const bytesPerRowAligned = Math.ceil((w * 8) / 256) * 256;
    const size = bytesPerRowAligned * h;
    const buf = this.ctx.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.ctx.device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: tex },
      { buffer: buf, bytesPerRow: bytesPerRowAligned, rowsPerImage: h },
      { width: w, height: h, depthOrArrayLayers: 1 },
    );
    this.ctx.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    buf.destroy();
    // Unpack: aligned rows -> contiguous w*h*2.
    const floatsPerRow = bytesPerRowAligned / 4;
    const out = new Float32Array(w * h * 2);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        out[(y * w + x) * 2 + 0] = raw[y * floatsPerRow + x * 2 + 0]!;
        out[(y * w + x) * 2 + 1] = raw[y * floatsPerRow + x * 2 + 1]!;
      }
    }
    return out;
  }

  /** Sum of all `h` values in the water texture, in m³ (assuming dx² area each). */
  async totalVolume(): Promise<number> {
    const w = await this.readWater();
    let sum = 0;
    const area = this.grid.dx * this.grid.dx;
    for (let i = 0; i < w.length; i += 2) sum += Math.max(0, w[i]!) * area;
    return sum;
  }

  destroy(): void {
    this.bedTex.destroy();
    this.waterTex.destroy();
    this.fluxLR.destroy();
    this.fluxUD.destroy();
    this.velocity.destroy();
    this.chunkId.destroy();
    this.chunkVel.destroy();
    this.boundary.destroy();
    this.prevBed.destroy();
    this.forceAccum.destroy();
    this.torqueAccum.destroy();
    this.hDelta.destroy();
    this.chunkCOMs.destroy();
    this.params.destroy();
    for (const b of this.readbackBuffers) b.destroy();
  }
}
