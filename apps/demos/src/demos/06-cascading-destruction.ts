import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { createStress, tickStress, FractureScheduler, BoundaryType } from 'isenflow';

/** Rho (kg/m^3) * g (m/s^2) * 0.5 — hydrostatic pressure constant. */
const HALF_RHO_G = 0.5 * 1000 * 9.81;

interface ChunkEntry {
  index: number;
  cells: { x: number; y: number; w: number; h: number };
  mesh: THREE.Mesh;
  stress: ReturnType<typeof createStress>;
  /** Stress budget (N·s) — staggered so upstream walls fail first. */
  budget: number;
}

const demo: Demo = {
  id: '06-cascading-destruction',
  label: 'F. Cascading Destruction',
  description:
    'Row of three walls along the flow path. Hydrostatic load builds; the upstream wall fails first; a fracture scheduler rate-limits to one wall per frame.',
  setup(ctx) {
    const g = ctx.solver.grid;
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x666666 });
    const chunks: ChunkEntry[] = [];
    const positions = [0.3, 0.5, 0.7].map((f) => Math.floor(g.width * f));
    const budgets = [25000, 85000, 170000];
    // On KP (SweSolver), use markSolidRegion so the 3 m walls are TRUE
    // impermeable barriers instead of "tall water that overtops" — which
    // is what the legacy VP scheme would do, producing wet/dry spike
    // artifacts as the tide rose against the wall.
    const supportsSolid =
      typeof (ctx.solver as { markSolidRegion?: unknown }).markSolidRegion === 'function';
    positions.forEach((px, idx) => {
      const region = { x: px, y: 0, w: 1, h: g.height };
      const wall = new Float32Array(g.height).fill(3);
      ctx.solver.writeBedRegion(region, wall);
      if (supportsSolid) {
        (ctx.solver as { markSolidRegion: (r: { x: number; y: number; w: number; h: number }) => void }).markSolidRegion(region);
      }
      const mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(g.dx, 3, g.height * g.dx), wallMat.clone()));
      mesh.name = `wall${idx}`;
      mesh.position.set(g.origin[0] + (px + 0.5) * g.dx, 1.5, g.origin[1] + (g.height * g.dx) / 2);
      ctx.scene.add(mesh);
      chunks.push({
        index: idx,
        cells: region,
        mesh,
        stress: createStress(budgets[idx]!, 0.92),
        budget: budgets[idx]!,
      });
    });

    // Inflow boundary on west edge — tide ramps via writeBoundaryRegionTarget
    ctx.solver.writeBoundaryRegionTarget(
      { x: 0, y: 0, w: 1, h: g.height },
      BoundaryType.Inflow,
      0,
    );
    // Open boundary on east edge
    ctx.solver.writeBoundaryRegionTarget(
      { x: g.width - 1, y: 0, w: 1, h: g.height },
      BoundaryType.Open,
      0,
    );

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.chunks = chunks;
    ctx.scratch.t = 0;
    ctx.scratch.scheduler = new FractureScheduler<ChunkEntry>(1);
    ctx.scratch.fracturedThisFrame = 0;
    ctx.scratch.fracturedCount = 0;
    ctx.scratch.tideH = 0;
  },
  tick(ctx: DemoContext, dt) {
    (ctx.scratch.water as WaterSurface).update();
    const g = ctx.solver.grid;
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    const tide = Math.min(3.5, 0.4 * (ctx.scratch.t as number));
    ctx.scratch.tideH = tide;
    // Ramp tide via boundary target depth
    ctx.solver.writeBoundaryRegionTarget(
      { x: 0, y: 0, w: 1, h: g.height },
      BoundaryType.Inflow,
      tide,
    );

    const chunks = ctx.scratch.chunks as ChunkEntry[];
    const scheduler = ctx.scratch.scheduler as FractureScheduler<ChunkEntry>;
    const wData = (ctx.scratch.water as WaterSurface).lastWaterData;
    for (const c of chunks) {
      if (!c.mesh.visible) continue;
      let F = 0;
      if (wData) {
        // Sample actual water depth one column upstream of the wall
        const upCol = Math.max(0, c.cells.x - 1);
        for (let j = 0; j < g.height; j++) {
          const h = wData[(j * g.width + upCol) * 2] ?? 0;
          F += HALF_RHO_G * h * h * g.dx;
        }
      }
      if (tickStress(c.stress, F, dt)) {
        scheduler.request(c, 1.0 / Math.max(1, c.index + 1));
      }
    }
    const drained = scheduler.drainFrame();
    ctx.scratch.fracturedThisFrame = drained.length;
    if (drained.length > 0) {
      ctx.scratch.fracturedCount = (ctx.scratch.fracturedCount as number) + drained.length;
      for (const c of drained) {
        c.mesh.visible = false;
        const blank = new Float32Array(c.cells.h).fill(0);
        ctx.solver.writeBedRegion(c.cells, blank);
        // Clear the Solid mask back to Interior so water can flow through
        ctx.solver.writeBoundaryRegion(c.cells, BoundaryType.Interior);
      }
    }
  },
};

export default demo;
