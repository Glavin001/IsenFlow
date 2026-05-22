/**
 * Headless test harness — invoked by Playwright (no renderer, no canvas).
 *
 * Exposes window.__isenflow_test with imperative entry points.
 */
import {
  acquireGPU,
  SimulationGrid,
  VirtualPipesSolver,
  HeightfieldRasterizer,
  ForceReadback,
  isWebGPUAvailable,
} from 'isenflow';

const log = (msg: string) => {
  const el = document.getElementById('log');
  if (el) el.textContent = `${el.textContent}\n${msg}`;
  // eslint-disable-next-line no-console
  console.log(msg);
};

async function runConservation(stepCount = 300): Promise<{ drift: number; initial: number; final: number; }> {
  const gpu = await acquireGPU();
  const grid = new SimulationGrid({ width: 64, height: 64, dx: 1, origin: [-32, -32], initialDepth: 1 });
  const solver = new VirtualPipesSolver(gpu, grid, { dt: 0.01, substepsPerFrame: 1 });
  const initial = await solver.totalVolume();
  for (let i = 0; i < stepCount; i++) solver.step();
  const final = await solver.totalVolume();
  solver.destroy();
  return { drift: Math.abs(final - initial) / Math.max(1e-6, initial), initial, final };
}

async function runDamBreak(
  steps = 600,
  initialDepth = 2,
  domainCells = 256,
): Promise<{ frontCellAtEnd: number; expectedCell: number; }> {
  const dx = 0.2;
  const gpu = await acquireGPU();
  const grid = new SimulationGrid({ width: domainCells, height: 8, dx, origin: [0, -1] });
  const solver = new VirtualPipesSolver(gpu, grid, { dt: 0.005, substepsPerFrame: 1 });
  const halfCells = Math.floor(domainCells / 4);
  solver.writeWaterRegion(
    { x: 0, y: 0, w: halfCells, h: 8 },
    new Float32Array(halfCells * 8).fill(initialDepth),
  );
  for (let s = 0; s < steps; s++) solver.step();
  const water = await solver.readWater();
  let frontCell = halfCells;
  for (let i = domainCells - 1; i >= 0; i--) {
    const h = water[(4 * domainCells + i) * 2] ?? 0;
    if (h > 0.01) { frontCell = i; break; }
  }
  const t = steps * 0.005;
  const expectedM = 2 * Math.sqrt(9.81 * initialDepth) * t;
  const expectedCell = halfCells + expectedM / dx;
  solver.destroy();
  return { frontCellAtEnd: frontCell, expectedCell };
}

/**
 * Force-accumulator round-trip: stamp a single chunk with non-zero relative
 * velocity vs the water column, run accumulateForces once, read back the
 * GPU forceAccum, and confirm a non-zero force on that chunk.
 */
async function runForceAccumulator(): Promise<{
  peakMagnitudeN: number;
  nonZeroChunks: number;
  fxN: number;
  fyN: number;
}> {
  const gpu = await acquireGPU();
  const grid = new SimulationGrid({ width: 32, height: 32, dx: 1, origin: [-16, -16], initialDepth: 1.0 });
  const solver = new VirtualPipesSolver(gpu, grid, { dt: 0.01, substepsPerFrame: 1, maxChunks: 4 });
  const rast = new HeightfieldRasterizer(solver);

  // Place a "boat" chunk in cells (10..18, 10..18) with strong west→east linvel
  // (the water is still, so relative motion is huge).
  const chunkId = 1;
  rast.bakeDynamicBodies([
    {
      chunkId,
      aabb: { minX: -6, minZ: -6, maxX: 2, maxZ: 2 },
      topY: 0.5,
      halfY: 0.5,
      linvel: [4.0, 0, 0],
      com: [-2, 0.0, -2],
    },
  ]);
  // Run a few SWE steps so velocity field exists, then accumulate.
  for (let i = 0; i < 4; i++) solver.step();
  solver.zeroForces();
  solver.accumulateForces();
  // Wait for queue to flush by reading the accumulator buffer.
  const i32 = await solver.readForceAccum();
  // Decode to ChunkForce[].
  let nonZero = 0;
  let peak = 0;
  let fxN = 0, fyN = 0;
  for (let c = 0; c < solver.maxChunks; c++) {
    const fx = (i32[c * 6 + 0] ?? 0) / 1e4;
    const fy = (i32[c * 6 + 1] ?? 0) / 1e4;
    const fz = (i32[c * 6 + 2] ?? 0) / 1e4;
    const mag = Math.hypot(fx, fy, fz);
    if (mag > 1e-3) nonZero++;
    if (mag > peak) peak = mag;
    if (c === chunkId) { fxN = fx; fyN = fy; }
  }
  solver.destroy();
  return { peakMagnitudeN: peak, nonZeroChunks: nonZero, fxN, fyN };
}

/**
 * Body→water displacement round-trip: stamp a bed rise (simulate a body that
 * just appeared in some cells), run applyDisplacement, verify total water
 * volume is conserved within 1% and that water near the rise increased.
 */
async function runDisplacement(): Promise<{
  volumeBefore: number;
  volumeAfter: number;
  driftPct: number;
  hMaxNeighborBefore: number;
  hMaxNeighborAfter: number;
}> {
  const gpu = await acquireGPU();
  const grid = new SimulationGrid({ width: 32, height: 32, dx: 1, origin: [0, 0], initialDepth: 1.0 });
  const solver = new VirtualPipesSolver(gpu, grid, { dt: 0.01, substepsPerFrame: 1 });

  // Snapshot prevBed so first applyDisplacement sees `delta = bed_now - bed_prev = 0`.
  solver.applyDisplacement();
  const volumeBefore = await solver.totalVolume();
  const water0 = await solver.readWater();
  const W = grid.width;
  const ci = 16, cj = 16;
  const neighbors = [[ci-1, cj], [ci+1, cj], [ci, cj-1], [ci, cj+1]];
  let hNeighBefore = 0;
  for (const [i, j] of neighbors) hNeighBefore = Math.max(hNeighBefore, water0[(j! * W + i!) * 2] ?? 0);

  // Raise bed at the centre cell by 0.5m (simulate body appearance) — only
  // the .y channel so the displacement kernel sees a positive delta.
  solver.writeBedTotalRegion({ x: ci, y: cj, w: 1, h: 1 }, new Float32Array([0.5]));
  // Run one displacement pass — this should shovel water into 8 neighbors.
  solver.applyDisplacement();

  const volumeAfter = await solver.totalVolume();
  const water1 = await solver.readWater();
  let hNeighAfter = 0;
  for (const [i, j] of neighbors) hNeighAfter = Math.max(hNeighAfter, water1[(j! * W + i!) * 2] ?? 0);

  solver.destroy();
  const driftPct = Math.abs(volumeAfter - volumeBefore) / Math.max(1e-6, volumeBefore);
  return {
    volumeBefore,
    volumeAfter,
    driftPct,
    hMaxNeighborBefore: hNeighBefore,
    hMaxNeighborAfter: hNeighAfter,
  };
}

async function probeAdapter(): Promise<{ available: boolean; reason?: string; adapter?: string }> {
  if (!isWebGPUAvailable()) return { available: false, reason: 'navigator.gpu missing' };
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { available: false, reason: 'requestAdapter returned null' };
    let info = '';
    try {
      const i = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
      if (i) info = `${i.vendor ?? '?'} / ${i.architecture ?? '?'} / ${i.device ?? '?'}`;
    } catch { /* ignore */ }
    const device = await adapter.requestDevice().catch(() => null);
    if (!device) return { available: false, reason: 'requestDevice returned null', adapter: info };
    device.destroy?.();
    return { available: true, adapter: info };
  } catch (err) {
    return { available: false, reason: (err as Error).message };
  }
}

window.__isenflow_test = {
  ready: false,
  runConservation,
  runDamBreak,
  hasWebGPU: isWebGPUAvailable,
  probeAdapter,
  runForceAccumulator: async () => {
    const r = await runForceAccumulator();
    return { peakMagnitude: r.peakMagnitudeN, nonZeroChunks: r.nonZeroChunks };
  },
  runDisplacement: async () => {
    const r = await runDisplacement();
    return { totalHDelta: r.hMaxNeighborAfter - r.hMaxNeighborBefore, volumeBefore: r.volumeBefore, volumeAfter: r.volumeAfter };
  },
};

// Reference imports so they aren't tree-shaken.
void ForceReadback;

log(`WebGPU feature flag: ${isWebGPUAvailable()}`);
probeAdapter().then((p) => log(`adapter probe: ${JSON.stringify(p)}`)).catch(() => {});
window.__isenflow_test.ready = true;
log('harness ready');
