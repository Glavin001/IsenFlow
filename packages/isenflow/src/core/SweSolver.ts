/**
 * Shallow-Water solver based on the **Kurganov–Petrova central-upwind**
 * scheme (KP07).  See `cpu/CpuKurganov.ts` for the CPU oracle this is
 * byte-verified against.
 *
 * Public API is intentionally compatible with the legacy `VirtualPipesSolver`
 * so demos and coupling code (rasterizer, force readback, displacement) can
 * switch transparently:
 *
 *   - same grid, params, boundary types, write* / read* helpers
 *   - same `step()`, `applyDisplacement()`, `accumulateForces()`,
 *     `zeroForces()`, `setCpuBuoyancyMode()` orchestration
 *   - exposes a `water` buffer with `(h, h_prev)` layout for consumers
 *     (renderer, ImpactDisplacement, etc.) that don't need momentum
 *   - additionally exposes `state` (h, hu, hv, _pad) for code that wants
 *     true momentum access
 *
 * Key differences from VP:
 *   - State is **conservative** (h, hu, hv) — momentum is primary, not
 *     reconstructed from flux differences.
 *   - SSP-RK2 (Heun) time integration: 2 stages per logical step.
 *   - Audusse hydrostatic reconstruction makes the scheme well-balanced
 *     and positivity-preserving — no `max(0.05, h)` cliffs.
 *   - `Solid` boundary type (6) is honoured for true impermeable walls.
 */

import type { GPUContext } from './GPUContext.js';
import type { SimulationGrid } from './SimulationGrid.js';
import { ShaderSource } from './shaders/index.js';

export interface SweSolverOptions {
  readonly dt: number;
  readonly substepsPerFrame: number;
  readonly gravity?: number;
  readonly maxChunks?: number;
  /** Manning roughness coefficient (default 0.03 for natural channels). */
  readonly manningN?: number;
  /** Kurganov desingularization regularizer (default 1e-3 m). */
  readonly desingEpsilon?: number;
}

const DEFAULT_OPTS: {
  gravity: number;
  maxChunks: number;
  manningN: number;
  desingEpsilon: number;
} = {
  gravity: 9.81,
  maxChunks: 256,
  manningN: 0.03,
  desingEpsilon: 1e-3,
};

const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class SweSolver {
  readonly ctx: GPUContext;
  readonly grid: SimulationGrid;
  readonly opts: SweSolverOptions & typeof DEFAULT_OPTS;

  // -- Authoritative conservative state on the GPU ---------------------------

  /** Conservative state: 4 floats per cell (h, hu, hv, _pad). */
  readonly state: GPUBuffer;
  /**
   * Output buffer for kp_update — ping-pongs with `state` to avoid the
   * data-race that would occur if kp_update both read and wrote `state`
   * (WGSL does not synchronize between workgroup invocations, so an
   * invocation reading a neighbor's state could see partial/stale data
   * being concurrently written by that neighbor's own invocation).
   */
  readonly stateNext: GPUBuffer;
  /** Snapshot of `state` at the start of each step (for RK2 averaging). */
  readonly stateSnap: GPUBuffer;
  /** Slopes: 8 floats per cell (dw_x, dw_y, dhu_x, dhu_y, dhv_x, dhv_y, maxSpeed, _pad). */
  readonly slopes: GPUBuffer;

  // -- Legacy-compatible views (kept for renderer + coupling) ----------------

  /**
   * Compatibility view: `water` exposes (h, h_prev) per cell exactly like
   * the old `VirtualPipesSolver.water` buffer.  Re-derived each step from
   * `state` so renderer / ImpactDisplacement / readWater() keep working
   * without modification.
   */
  readonly water: GPUBuffer;
  /**
   * Compatibility view: `velocity` exposes (u, v) per cell, derived from
   * (hu, hv) with Kurganov desingularization once per step.  Same layout
   * as the old `VirtualPipesSolver.velocity` buffer.
   */
  readonly velocity: GPUBuffer;

  // -- Scheme-independent infrastructure (shared with legacy code) -----------

  readonly bed: GPUBuffer;
  readonly chunkId: GPUBuffer;
  readonly chunkVel: GPUBuffer;
  readonly boundary: GPUBuffer;
  readonly boundaryTargetH: GPUBuffer;
  readonly prevBed: GPUBuffer;
  readonly hDelta: GPUBuffer;
  readonly forceAccum: GPUBuffer;
  readonly chunkCOMs: GPUBuffer;

  readonly params: GPUBuffer;
  /** Stage-1 uniform (α=0, β=1, γ=1)         — written once at construction. */
  readonly stage1Params: GPUBuffer;
  /** Stage-2 uniform (α=½, β=½, γ=½, manning) — written once at construction. */
  readonly stage2Params: GPUBuffer;
  private paramsCpu: ArrayBuffer;
  private paramsView: DataView;

  readonly terrainMirror: Float32Array;

  private pipelines: Record<string, GPUComputePipeline> = {};
  private cachedBindGroups: Record<string, GPUBindGroup> = {};

  constructor(ctx: GPUContext, grid: SimulationGrid, opts: SweSolverOptions) {
    this.ctx = ctx;
    this.grid = grid;
    this.opts = { ...DEFAULT_OPTS, ...opts };
    const dev = ctx.device;
    const cells = grid.width * grid.height;

    // KP state: vec4 per cell — TWO state buffers for ping-pong so kp_update
    // never reads and writes the same buffer (would race in WGSL).
    this.state     = dev.createBuffer({ size: cells * 16, usage: STORAGE_USAGE, label: 'sweKP.state' });
    this.stateNext = dev.createBuffer({ size: cells * 16, usage: STORAGE_USAGE, label: 'sweKP.stateNext' });
    this.stateSnap = dev.createBuffer({ size: cells * 16, usage: STORAGE_USAGE, label: 'sweKP.stateSnap' });
    this.slopes    = dev.createBuffer({ size: cells * 32, usage: STORAGE_USAGE, label: 'sweKP.slopes' });

    // Legacy-compatible views derived from state
    this.water     = dev.createBuffer({ size: cells * 2 * 4, usage: STORAGE_USAGE, label: 'sweKP.water' });
    this.velocity  = dev.createBuffer({ size: cells * 2 * 4, usage: STORAGE_USAGE, label: 'sweKP.velocity' });

    // Scheme-independent buffers (same layout as VP for compatibility)
    this.bed       = dev.createBuffer({ size: cells * 2 * 4, usage: STORAGE_USAGE, label: 'sweKP.bed' });
    this.chunkId   = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'sweKP.chunkId' });
    this.chunkVel  = dev.createBuffer({ size: cells * 4 * 4, usage: STORAGE_USAGE, label: 'sweKP.chunkVel' });
    this.boundary  = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'sweKP.boundary' });
    this.boundaryTargetH = dev.createBuffer({ size: cells * 4, usage: STORAGE_USAGE, label: 'sweKP.boundaryTargetH' });
    this.prevBed   = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'sweKP.prevBed' });
    this.hDelta    = dev.createBuffer({ size: cells * 4,     usage: STORAGE_USAGE, label: 'sweKP.hDelta' });

    const chunkBytes = this.opts.maxChunks * 6 * 4;
    this.forceAccum = dev.createBuffer({ size: chunkBytes, usage: STORAGE_USAGE, label: 'sweKP.forceAccum' });
    this.chunkCOMs  = dev.createBuffer({
      size: Math.max(256, this.opts.maxChunks) * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'sweKP.chunkCOMs',
    });

    this.terrainMirror = new Float32Array(cells);

    // SimParams (re-use legacy layout for compatibility with shared kernels):
    //  0..3 width(u32), 4..7 height(u32), 8..11 dx, 12..15 dt, 16..19 gravity,
    //  20..23 damping (unused in KP), 24..27 pipeArea (unused), 28..31 pipeLen (unused),
    //  32..35 manningN, 36..39 originX, 40..43 originZ, 44..47 cpuBuoyancyMode
    this.paramsCpu = new ArrayBuffer(48);
    this.paramsView = new DataView(this.paramsCpu);
    this.params = dev.createBuffer({
      size: this.paramsCpu.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'sweKP.params',
    });

    // KpStageParams: 32 bytes (8 × f32) — alpha, beta, gamma, applyManning, _pad×4
    // Two separate buffers — one per RK2 stage — so we can record BOTH stages
    // into a single command buffer and submit them in one go.  Avoids the
    // 3-submits-per-step overhead that previously cost ~10% of frame budget
    // on Apple Metal.
    this.stage1Params = dev.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'sweKP.stage1Params',
    });
    this.stage2Params = dev.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'sweKP.stage2Params',
    });
    {
      const buf = new ArrayBuffer(32);
      const dv = new DataView(buf);
      // Stage 1: U^(1) = U^n + Δt·L(U^n)   →  α=0, β=1, γ=1, manning=0
      dv.setFloat32(0, 0, true); dv.setFloat32(4, 1, true);
      dv.setFloat32(8, 1, true); dv.setFloat32(12, 0, true);
      this.ctx.queue.writeBuffer(this.stage1Params, 0, buf);
      // Stage 2: U^{n+1} = ½U^n + ½(U^(1)+Δt·L)  →  α=½, β=½, γ=½, manning=1
      dv.setFloat32(0, 0.5, true); dv.setFloat32(4, 0.5, true);
      dv.setFloat32(8, 0.5, true); dv.setFloat32(12, 1, true);
      this.ctx.queue.writeBuffer(this.stage2Params, 0, buf);
    }

    this.uploadParams();
    this.seedInitial();
    this.buildPipelines();
    this.buildBindGroups();
  }

  get forceAccumStrideBytes(): number { return 6 * 4; }
  get forceAccumBytes(): number { return this.opts.maxChunks * this.forceAccumStrideBytes; }
  get maxChunks(): number { return this.opts.maxChunks; }

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
    dv.setFloat32(20, 1.0, true);  // damping (legacy, unused in KP)
    dv.setFloat32(24, this.grid.dx * this.grid.dx, true); // pipeArea (legacy)
    dv.setFloat32(28, this.grid.dx, true);                // pipeLen (legacy)
    dv.setFloat32(32, this.opts.manningN, true);
    dv.setFloat32(36, this.grid.origin[0], true);
    dv.setFloat32(40, this.grid.origin[1], true);
    dv.setFloat32(44, 0, true); // cpuBuoyancyMode
    this.ctx.queue.writeBuffer(this.params, 0, this.paramsCpu);
  }

  private seedInitial(): void {
    const cells = this.grid.cells;

    // Bed: (terrain, total) interleaved
    const bedData = new Float32Array(cells * 2);
    for (let i = 0; i < cells; i++) {
      const b = this.grid.bedSeed[i] ?? 0;
      bedData[i * 2] = b;
      bedData[i * 2 + 1] = b;
      this.terrainMirror[i] = b;
    }
    this.ctx.queue.writeBuffer(this.bed, 0, bedData);

    // State: (h, hu, hv, _pad) per cell, initial h = initialDepth
    const stateData = new Float32Array(cells * 4);
    if (this.grid.initialDepth > 0) {
      for (let i = 0; i < cells; i++) {
        stateData[i * 4] = this.grid.initialDepth;
      }
    }
    this.ctx.queue.writeBuffer(this.state, 0, stateData);

    // Mirror initial state into `water` (h, h_prev) compatibility view
    if (this.grid.initialDepth > 0) {
      const waterData = new Float32Array(cells * 2);
      for (let i = 0; i < cells; i++) {
        waterData[i * 2] = this.grid.initialDepth;
        waterData[i * 2 + 1] = this.grid.initialDepth;
      }
      this.ctx.queue.writeBuffer(this.water, 0, waterData);
    }

    // prevBed seed = total bed
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
    this.pipelines.kpSlopes = make('kpSlopes', ShaderSource.kpSlopes);
    this.pipelines.kpUpdate = make('kpUpdate', ShaderSource.kpUpdate);
    this.pipelines.kpRefreshViews = make('kpRefreshViews', ShaderSource.kpRefreshViews);
    this.pipelines.kpSyncState = make('kpSyncState', ShaderSource.kpSyncState);
    this.pipelines.displace = make('displace', ShaderSource.applyDisplacement);
    this.pipelines.foldDelta = make('foldDelta', ShaderSource.foldHDelta);
    this.pipelines.snapshot = make('snapshot', ShaderSource.snapshotBed);
    this.pipelines.accumulate = make('accumulate', ShaderSource.accumulateForces);
    this.pipelines.zeroForces = make('zeroForces', ShaderSource.zeroForces);
  }

  private buildBindGroups(): void {
    const dev = this.ctx.device;
    const bg = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) =>
      dev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

    // Two slope bind groups so each RK2 stage's slopes are computed from
    // the correct state buffer (stage 1 from `state`, stage 2 from `stateNext`).
    this.cachedBindGroups.kpSlopes1 = bg(this.pipelines.kpSlopes!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.state } },
      { binding: 2, resource: { buffer: this.bed } },
      { binding: 3, resource: { buffer: this.slopes } },
    ]);
    this.cachedBindGroups.kpSlopes2 = bg(this.pipelines.kpSlopes!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.stateNext } },
      { binding: 2, resource: { buffer: this.bed } },
      { binding: 3, resource: { buffer: this.slopes } },
    ]);

    // Ping-pong: stage 1 reads from `state` (= U^n), writes to `stateNext` (= U^(1)).
    this.cachedBindGroups.kpUpdate1 = bg(this.pipelines.kpUpdate!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.stage1Params } },
      { binding: 2, resource: { buffer: this.state } },       // state_in
      { binding: 3, resource: { buffer: this.stateNext } },   // state_out
      { binding: 4, resource: { buffer: this.stateSnap } },
      { binding: 5, resource: { buffer: this.bed } },
      { binding: 6, resource: { buffer: this.slopes } },
      { binding: 7, resource: { buffer: this.boundary } },
      { binding: 8, resource: { buffer: this.boundaryTargetH } },
    ]);

    // Ping-pong: stage 2 reads from `stateNext` (= U^(1)), writes to `state` (= U^(n+1)).
    this.cachedBindGroups.kpUpdate2 = bg(this.pipelines.kpUpdate!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.stage2Params } },
      { binding: 2, resource: { buffer: this.stateNext } },   // state_in
      { binding: 3, resource: { buffer: this.state } },       // state_out
      { binding: 4, resource: { buffer: this.stateSnap } },
      { binding: 5, resource: { buffer: this.bed } },
      { binding: 6, resource: { buffer: this.slopes } },
      { binding: 7, resource: { buffer: this.boundary } },
      { binding: 8, resource: { buffer: this.boundaryTargetH } },
    ]);

    this.cachedBindGroups.kpRefreshViews = bg(this.pipelines.kpRefreshViews!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.state } },
      { binding: 2, resource: { buffer: this.water } },
      { binding: 3, resource: { buffer: this.velocity } },
      { binding: 4, resource: { buffer: this.stateSnap } },
    ]);

    this.cachedBindGroups.kpSyncState = bg(this.pipelines.kpSyncState!, [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.water } },
      { binding: 2, resource: { buffer: this.state } },
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

  /**
   * Advance one logical SWE step using SSP-RK2 (Heun) integration.
   *
   * Records ALL passes (snapshot copy + 2 stages × 2 dispatches + refresh)
   * into a single command buffer and submits once.  Two separate stage
   * uniform buffers (`stage1Params`, `stage2Params`) are pre-populated at
   * construction, so the kernel reads the correct (α, β, γ, manning) on
   * each pass without needing inter-submit uniform writes.
   */
  step(): void {
    const gx = Math.ceil(this.grid.width / 8);
    const gy = Math.ceil(this.grid.height / 8);
    const encoder = this.ctx.device.createCommandEncoder({ label: 'sweKP.step' });

    for (let s = 0; s < this.opts.substepsPerFrame; s++) {
      // Snapshot U^n → stateSnap.  After this, state still holds U^n.
      encoder.copyBufferToBuffer(this.state, 0, this.stateSnap, 0, this.state.size);

      // -------- Stage 1: U^(1) = U^n + Δt·L(U^n) --------
      //   slopes computed from state (= U^n); update reads state, writes stateNext.
      {
        const pass = encoder.beginComputePass({ label: 'kpSlopes-1' });
        pass.setPipeline(this.pipelines.kpSlopes!);
        pass.setBindGroup(0, this.cachedBindGroups.kpSlopes1!);
        pass.dispatchWorkgroups(gx, gy, 1);
        pass.end();
      }
      {
        const pass = encoder.beginComputePass({ label: 'kpUpdate-1' });
        pass.setPipeline(this.pipelines.kpUpdate!);
        pass.setBindGroup(0, this.cachedBindGroups.kpUpdate1!);
        pass.dispatchWorkgroups(gx, gy, 1);
        pass.end();
      }

      // -------- Stage 2: U^{n+1} = ½U^n + ½(U^(1) + Δt·L(U^(1))) --------
      //   slopes computed from stateNext (= U^(1)); update reads stateNext, writes state.
      {
        const pass = encoder.beginComputePass({ label: 'kpSlopes-2' });
        pass.setPipeline(this.pipelines.kpSlopes!);
        pass.setBindGroup(0, this.cachedBindGroups.kpSlopes2!);
        pass.dispatchWorkgroups(gx, gy, 1);
        pass.end();
      }
      {
        const pass = encoder.beginComputePass({ label: 'kpUpdate-2' });
        pass.setPipeline(this.pipelines.kpUpdate!);
        pass.setBindGroup(0, this.cachedBindGroups.kpUpdate2!);
        pass.dispatchWorkgroups(gx, gy, 1);
        pass.end();
      }
    }

    // Refresh the legacy (h, h_prev) and (u, v) views from conservative
    // state so renderer + force kernels see fresh values without any
    // round-trip.  Single dispatch in the same submit.
    {
      const pass = encoder.beginComputePass({ label: 'kpRefreshViews' });
      pass.setPipeline(this.pipelines.kpRefreshViews!);
      pass.setBindGroup(0, this.cachedBindGroups.kpRefreshViews!);
      pass.dispatchWorkgroups(gx, gy, 1);
      pass.end();
    }

    this.ctx.queue.submit([encoder.finish()]);
  }

  /**
   * Body→water displacement.  The displace+foldDelta passes modify the
   * legacy `water` buffer based on bed changes from rasterized dynamic
   * bodies.  KP's authoritative state is `state`, so we end with a
   * kpSyncState pass that copies water.x → state.x — without this, the
   * displacement is silently lost on the next step (since kp_update
   * reads from `state`, not from `water`).
   */
  applyDisplacement(): void {
    const encoder = this.ctx.device.createCommandEncoder({ label: 'sweKP.displace' });
    const gx = Math.ceil(this.grid.width / 8);
    const gy = Math.ceil(this.grid.height / 8);

    const passD = encoder.beginComputePass({ label: 'displace' });
    passD.setPipeline(this.pipelines.displace!);
    passD.setBindGroup(0, this.cachedBindGroups.displace!);
    passD.dispatchWorkgroups(gx, gy, 1);
    passD.end();

    const passF = encoder.beginComputePass({ label: 'foldDelta' });
    passF.setPipeline(this.pipelines.foldDelta!);
    passF.setBindGroup(0, this.cachedBindGroups.foldDelta!);
    passF.dispatchWorkgroups(gx, gy, 1);
    passF.end();

    const passS = encoder.beginComputePass({ label: 'snapshot' });
    passS.setPipeline(this.pipelines.snapshot!);
    passS.setBindGroup(0, this.cachedBindGroups.snapshot!);
    passS.dispatchWorkgroups(gx, gy, 1);
    passS.end();

    // KP-specific: push the displacement-induced h change back into the
    // conservative state buffer.  Without this, the next solver step
    // would not see the body's effect on the water surface.
    const passSync = encoder.beginComputePass({ label: 'kpSyncState' });
    passSync.setPipeline(this.pipelines.kpSyncState!);
    passSync.setBindGroup(0, this.cachedBindGroups.kpSyncState!);
    passSync.dispatchWorkgroups(gx, gy, 1);
    passSync.end();

    this.ctx.queue.submit([encoder.finish()]);
  }

  zeroForces(): void {
    const elements = this.opts.maxChunks * 6;
    const groups = Math.ceil(elements / 64);
    const encoder = this.ctx.device.createCommandEncoder({ label: 'sweKP.zeroForces' });
    const pass = encoder.beginComputePass({ label: 'zeroForces' });
    pass.setPipeline(this.pipelines.zeroForces!);
    pass.setBindGroup(0, this.cachedBindGroups.zeroForces!);
    pass.dispatchWorkgroups(groups, 1, 1);
    pass.end();
    this.ctx.queue.submit([encoder.finish()]);
  }

  accumulateForces(opts?: {
    copyTo?: GPUBuffer;
    onEncoder?: (encoder: GPUCommandEncoder) => void;
  }): void {
    const gx = Math.ceil(this.grid.width / 8);
    const gy = Math.ceil(this.grid.height / 8);
    const encoder = this.ctx.device.createCommandEncoder({ label: 'sweKP.accumulate' });
    const pass = encoder.beginComputePass({ label: 'accumulate' });
    pass.setPipeline(this.pipelines.accumulate!);
    pass.setBindGroup(0, this.cachedBindGroups.accumulate!);
    pass.dispatchWorkgroups(gx, gy, 1);
    pass.end();

    if (opts?.copyTo) {
      encoder.copyBufferToBuffer(this.forceAccum, 0, opts.copyTo, 0, this.forceAccumBytes);
    }
    if (opts?.onEncoder) opts.onEncoder(encoder);

    this.ctx.queue.submit([encoder.finish()]);
  }

  // ------------------------- Write helpers ---------------------------------

  writeBedFull(values: Float32Array): void {
    if (values.length !== this.grid.cells) throw new Error('writeBedFull: length');
    const packed = new Float32Array(this.grid.cells * 2);
    for (let i = 0; i < values.length; i++) {
      packed[i * 2] = values[i]!;
      packed[i * 2 + 1] = values[i]!;
      this.terrainMirror[i] = values[i]!;
    }
    this.ctx.queue.writeBuffer(this.bed, 0, packed);
  }

  writeBedRegion(region: { x: number; y: number; w: number; h: number }, values: Float32Array): void {
    if (values.length !== region.w * region.h) throw new Error('writeBedRegion: length');
    const W = this.grid.width;
    for (let row = 0; row < region.h; row++) {
      const rowOffsetCells = (region.y + row) * W + region.x;
      const packed = new Float32Array(region.w * 2);
      for (let i = 0; i < region.w; i++) {
        const v = values[row * region.w + i]!;
        packed[i * 2] = v;
        packed[i * 2 + 1] = v;
        this.terrainMirror[rowOffsetCells + i] = v;
      }
      this.ctx.queue.writeBuffer(this.bed, rowOffsetCells * 2 * 4, packed.buffer, packed.byteOffset, region.w * 2 * 4);
    }
  }

  writeBedTotalRegion(region: { x: number; y: number; w: number; h: number }, values: Float32Array): void {
    if (values.length !== region.w * region.h) throw new Error('writeBedTotalRegion: length');
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
    if (values.length !== this.grid.cells) throw new Error('writeWaterFull: length');
    // Update conservative state: h = value, hu = hv = 0 (no implied momentum)
    const stateData = new Float32Array(this.grid.cells * 4);
    for (let i = 0; i < values.length; i++) {
      stateData[i * 4] = values[i]!;
    }
    this.ctx.queue.writeBuffer(this.state, 0, stateData);
    // Also update legacy view
    const packed = new Float32Array(this.grid.cells * 2);
    for (let i = 0; i < values.length; i++) {
      packed[i * 2] = values[i]!;
      packed[i * 2 + 1] = values[i]!;
    }
    this.ctx.queue.writeBuffer(this.water, 0, packed);
  }

  writeWaterRegion(region: { x: number; y: number; w: number; h: number }, values: Float32Array): void {
    if (values.length !== region.w * region.h) throw new Error('writeWaterRegion: length');
    const W = this.grid.width;
    for (let row = 0; row < region.h; row++) {
      const rowOffsetCells = (region.y + row) * W + region.x;
      // Update state h channel (clear momentum in written cells)
      const statePacked = new Float32Array(region.w * 4);
      for (let i = 0; i < region.w; i++) {
        statePacked[i * 4] = values[row * region.w + i]!;
      }
      this.ctx.queue.writeBuffer(this.state, rowOffsetCells * 16, statePacked.buffer, statePacked.byteOffset, region.w * 16);
      // Mirror to legacy water view
      const waterPacked = new Float32Array(region.w * 2);
      for (let i = 0; i < region.w; i++) {
        const v = values[row * region.w + i]!;
        waterPacked[i * 2] = v;
        waterPacked[i * 2 + 1] = v;
      }
      this.ctx.queue.writeBuffer(this.water, rowOffsetCells * 2 * 4, waterPacked.buffer, waterPacked.byteOffset, region.w * 2 * 4);
    }
  }

  writeWaterCell(i: number, j: number, depth: number): void {
    const stateData = new Float32Array([depth, 0, 0, 0]);
    const stateOffset = (j * this.grid.width + i) * 16;
    this.ctx.queue.writeBuffer(this.state, stateOffset, stateData);

    const waterData = new Float32Array([depth, depth]);
    const waterOffset = (j * this.grid.width + i) * 2 * 4;
    this.ctx.queue.writeBuffer(this.water, waterOffset, waterData);
  }

  writeChunkIdRegion(region: { x: number; y: number; w: number; h: number }, chunkId: number): void {
    const W = this.grid.width;
    const row = new Uint32Array(region.w).fill(chunkId);
    for (let r = 0; r < region.h; r++) {
      const rowOffsetCells = (region.y + r) * W + region.x;
      this.ctx.queue.writeBuffer(this.chunkId, rowOffsetCells * 4, row.buffer, row.byteOffset, region.w * 4);
    }
  }

  writeChunkVelRegion(region: { x: number; y: number; w: number; h: number }, vx: number, vy: number, vz: number): void {
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
      this.ctx.queue.writeBuffer(this.chunkVel, rowOffsetCells * 16, row.buffer, row.byteOffset, region.w * 16);
    }
  }

  resetSimState(): void {
    const cells = this.grid.cells;
    const zero4 = new Float32Array(cells * 4);
    const zero8 = new Float32Array(cells * 8);
    const zero2 = new Float32Array(cells * 2);
    this.ctx.queue.writeBuffer(this.state, 0, zero4);
    this.ctx.queue.writeBuffer(this.stateNext, 0, zero4);
    this.ctx.queue.writeBuffer(this.stateSnap, 0, zero4);
    this.ctx.queue.writeBuffer(this.slopes, 0, zero8);
    this.ctx.queue.writeBuffer(this.water, 0, zero2);
    this.ctx.queue.writeBuffer(this.velocity, 0, zero2);
    this.clearChunkIds();
    this.clearChunkVel();
    this.setCpuBuoyancyMode(false);
  }

  clearChunkIds(): void {
    const zeros = new Uint32Array(this.grid.cells);
    this.ctx.queue.writeBuffer(this.chunkId, 0, zeros);
  }

  clearChunkVel(): void {
    const zeros = new Float32Array(this.grid.cells * 4);
    this.ctx.queue.writeBuffer(this.chunkVel, 0, zeros);
  }

  writeChunkCOMs(data: Float32Array): void {
    const required = this.opts.maxChunks * 4;
    const padded = data.length === required ? data : (() => {
      const p = new Float32Array(required);
      p.set(data.subarray(0, Math.min(data.length, required)));
      return p;
    })();
    this.ctx.queue.writeBuffer(this.chunkCOMs, 0, padded.buffer, padded.byteOffset, padded.byteLength);
  }

  writeChunkCOM(chunkId: number, x: number, y: number, z: number, halfY = 0): void {
    if (chunkId < 0 || chunkId >= this.opts.maxChunks) {
      throw new Error(`writeChunkCOM: chunkId ${chunkId} out of range [0, ${this.opts.maxChunks})`);
    }
    const arr = new Float32Array([x, y, z, halfY]);
    this.ctx.queue.writeBuffer(this.chunkCOMs, chunkId * 16, arr);
  }

  writeBoundaryRegion(region: { x: number; y: number; w: number; h: number }, boundaryType: number): void {
    const W = this.grid.width;
    const row = new Uint32Array(region.w).fill(boundaryType);
    for (let r = 0; r < region.h; r++) {
      const rowOffsetCells = (region.y + r) * W + region.x;
      this.ctx.queue.writeBuffer(this.boundary, rowOffsetCells * 4, row.buffer, row.byteOffset, region.w * 4);
    }
  }

  writeBoundaryRegionTarget(region: { x: number; y: number; w: number; h: number }, boundaryType: number, targetDepth: number): void {
    this.writeBoundaryRegion(region, boundaryType);
    const W = this.grid.width;
    const row = new Float32Array(region.w).fill(targetDepth);
    for (let r = 0; r < region.h; r++) {
      const rowOffsetCells = (region.y + r) * W + region.x;
      this.ctx.queue.writeBuffer(this.boundaryTargetH, rowOffsetCells * 4, row.buffer, row.byteOffset, region.w * 4);
    }
  }

  /** Mark cells as Solid (impermeable wall) — zero flux through any face touching them. */
  markSolidRegion(region: { x: number; y: number; w: number; h: number }): void {
    this.writeBoundaryRegion(region, 6);  // BoundaryType.Solid
    // Zero out h, hu, hv in those cells
    const stateZero = new Float32Array(region.w * region.h * 4);
    this.writeStateRegionRaw(region, stateZero);
  }

  /**
   * Write a region of the conservative state (h, hu, hv, _pad) per cell.
   * `state` is row-major, 4 floats per cell.  Used by KP-aware callers
   * like `applyImpact` to inject crater + ring + outward momentum in
   * one go, so KP radiates a true wave train physically.
   *
   * Writes are issued per-row (one writeBuffer per row), so the cost is
   * O(region.h) GPU buffer writes — efficient even for large regions.
   *
   * Also mirrors h into the legacy `water` buffer so consumers reading
   * `water` before the next step's refresh-views pass see the new h.
   */
  writeStateRegion(
    region: { x: number; y: number; w: number; h: number },
    state: Float32Array,
  ): void {
    if (state.length !== region.w * region.h * 4) {
      throw new Error('writeStateRegion: length mismatch');
    }
    this.writeStateRegionRaw(region, state);
    // Mirror h into the legacy water view
    const water = new Float32Array(region.w * region.h * 2);
    for (let i = 0; i < region.w * region.h; i++) {
      water[i * 2 + 0] = state[i * 4 + 0]!;
      water[i * 2 + 1] = state[i * 4 + 0]!;
    }
    const W = this.grid.width;
    for (let row = 0; row < region.h; row++) {
      const rowOffsetCells = (region.y + row) * W + region.x;
      const slice = water.subarray(row * region.w * 2, (row + 1) * region.w * 2);
      this.ctx.queue.writeBuffer(
        this.water,
        rowOffsetCells * 2 * 4,
        slice.buffer,
        slice.byteOffset,
        slice.byteLength,
      );
    }
  }

  /** Internal: stamp raw state values (vec4 per cell) into a region. */
  private writeStateRegionRaw(region: { x: number; y: number; w: number; h: number }, values: Float32Array): void {
    if (values.length !== region.w * region.h * 4) throw new Error('writeStateRegionRaw: length');
    const W = this.grid.width;
    for (let row = 0; row < region.h; row++) {
      const rowOffsetCells = (region.y + row) * W + region.x;
      const slice = values.subarray(row * region.w * 4, (row + 1) * region.w * 4);
      this.ctx.queue.writeBuffer(this.state, rowOffsetCells * 16, slice.buffer, slice.byteOffset, slice.byteLength);
    }
  }

  // ------------------------- Readback ---------------------------------------

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

  /**
   * Returns length width*height*2 [h, h_prev] interleaved (legacy compat).
   * Synthesizes (h, h_prev) by reading the conservative state and pairing
   * h with itself for h_prev.  Real h_prev temporal tracking is not yet
   * implemented in KP; consumers that care can poll twice and diff.
   */
  async readWater(): Promise<Float32Array> {
    const s = await this.readF32Buffer(this.state);
    const cells = this.grid.cells;
    const out = new Float32Array(cells * 2);
    for (let i = 0; i < cells; i++) {
      out[i * 2] = s[i * 4]!;
      out[i * 2 + 1] = s[i * 4]!;
    }
    return out;
  }

  /** Returns length width*height*2 [terrain, total] interleaved. */
  readBed(): Promise<Float32Array> { return this.readF32Buffer(this.bed); }

  /**
   * Returns length width*height*2 [u, v] interleaved.  Velocities are
   * derived from the conservative momentum via Kurganov desingularization
   * (the truly-accurate momentum field that boats / debris feel).
   */
  async readVelocity(): Promise<Float32Array> {
    const s = await this.readF32Buffer(this.state);
    const cells = this.grid.cells;
    const eps = this.opts.desingEpsilon;
    const eps4 = eps * eps * eps * eps;
    const SQRT2 = Math.SQRT2;
    const out = new Float32Array(cells * 2);
    for (let i = 0; i < cells; i++) {
      const h = s[i * 4]!;
      const hu = s[i * 4 + 1]!;
      const hv = s[i * 4 + 2]!;
      if (h <= 0) { out[i * 2] = 0; out[i * 2 + 1] = 0; continue; }
      const h2 = h * h;
      const h4 = h2 * h2;
      const denom = Math.sqrt(h4 + Math.max(h4, eps4));
      out[i * 2 + 0] = (SQRT2 * h * hu) / denom;
      out[i * 2 + 1] = (SQRT2 * h * hv) / denom;
    }
    return out;
  }

  /** Returns length width*height*4 [h, hu, hv, _pad] interleaved (KP-native). */
  readState(): Promise<Float32Array> { return this.readF32Buffer(this.state); }

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
    const s = await this.readState();
    let sum = 0;
    const area = this.grid.dx * this.grid.dx;
    for (let i = 0; i < s.length; i += 4) sum += Math.max(0, s[i]!) * area;
    return sum;
  }

  destroy(): void {
    for (const b of [
      this.state, this.stateNext, this.stateSnap, this.slopes,
      this.bed, this.water, this.velocity,
      this.chunkId, this.chunkVel, this.boundary, this.boundaryTargetH,
      this.prevBed, this.hDelta,
      this.forceAccum, this.chunkCOMs, this.params, this.stage1Params, this.stage2Params,
    ]) {
      b.destroy();
    }
  }
}
