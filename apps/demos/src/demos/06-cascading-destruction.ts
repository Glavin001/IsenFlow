import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { createStress, tickStress, FractureScheduler, BoundaryType } from 'isenflow';

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
    positions.forEach((px, idx) => {
      const region = { x: px, y: 0, w: 1, h: g.height };
      const wall = new Float32Array(g.height).fill(3);
      ctx.solver.writeBedRegion(region, wall);
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

    const STRESS_DT = 1 / 60;
    const stressTicks = Math.max(1, Math.round(dt / STRESS_DT));
    const chunks = ctx.scratch.chunks as ChunkEntry[];
    const scheduler = ctx.scratch.scheduler as FractureScheduler<ChunkEntry>;
    for (const c of chunks) {
      if (!c.mesh.visible) continue;
      const F = 0.5 * 1000 * 9.81 * tide * tide * g.height * g.dx;
      for (let s = 0; s < stressTicks; s++) {
        if (tickStress(c.stress, F, STRESS_DT)) {
          scheduler.request(c, 1.0 / Math.max(1, c.index + 1));
          break;
        }
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
      }
    }
  },
};

export default demo;
