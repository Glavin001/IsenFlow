/**
 * Headless test harness — invoked by Playwright (no renderer, no canvas).
 *
 * Exposes window.__isenflow_test with imperative entry points.
 */
import {
  acquireGPU,
  SimulationGrid,
  VirtualPipesSolver,
  isWebGPUAvailable,
} from 'isenflow';

const log = (msg: string) => {
  const el = document.getElementById('log');
  if (el) el.textContent = `${el.textContent}\n${msg}`;
  console.log(msg);
};

async function runConservation(stepCount = 300): Promise<{ drift: number; initial: number; final: number; }> {
  const gpu = await acquireGPU();
  const grid = new SimulationGrid({ width: 64, height: 64, dx: 1, origin: [-32, -32], initialDepth: 1 });
  const solver = new VirtualPipesSolver(gpu, grid, { dt: 0.01, substepsPerFrame: 1 });
  const initial = await solver.totalVolume();
  for (let i = 0; i < stepCount; i++) solver.step();
  // Drain pending GPU work via a readback round-trip.
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
  // Seed left half with column of water.
  const data = new Float32Array(domainCells * 8 * 2);
  const halfCells = Math.floor(domainCells / 4);
  for (let j = 0; j < 8; j++) {
    for (let i = 0; i < halfCells; i++) {
      data[(j * domainCells + i) * 2] = initialDepth;
      data[(j * domainCells + i) * 2 + 1] = initialDepth;
    }
  }
  gpu.queue.writeTexture(
    { texture: solver.waterTex }, data,
    { bytesPerRow: domainCells * 8, rowsPerImage: 8 },
    { width: domainCells, height: 8, depthOrArrayLayers: 1 },
  );
  for (let s = 0; s < steps; s++) solver.step();
  const water = await solver.readWater();
  // Find rightmost wet cell along row 4.
  let frontCell = halfCells;
  for (let i = domainCells - 1; i >= 0; i--) {
    const h = water[(4 * domainCells + i) * 2] ?? 0;
    if (h > 0.01) { frontCell = i; break; }
  }
  const t = steps * 0.005;
  // Wave-front travels at 2√(g·H₀) from the original dam location.
  const expectedM = 2 * Math.sqrt(9.81 * initialDepth) * t;
  const expectedCell = halfCells + expectedM / dx;
  solver.destroy();
  return { frontCellAtEnd: frontCell, expectedCell };
}

declare global {
  interface Window {
    __isenflow_test: {
      ready: boolean;
      runConservation: typeof runConservation;
      runDamBreak: typeof runDamBreak;
      hasWebGPU: () => boolean;
    };
  }
}

window.__isenflow_test = {
  ready: false,
  runConservation,
  runDamBreak,
  hasWebGPU: isWebGPUAvailable,
};

log(`WebGPU available: ${isWebGPUAvailable()}`);
window.__isenflow_test.ready = true;
log('harness ready');
