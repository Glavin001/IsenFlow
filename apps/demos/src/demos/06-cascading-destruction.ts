import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { createStress, tickStress } from 'isenflow';

interface ChunkEntry {
  cells: { x: number; y: number; w: number; h: number };
  topY: number;
  mesh: THREE.Mesh;
  stress: ReturnType<typeof createStress>;
}

const demo: Demo = {
  id: '06-cascading-destruction',
  label: 'F. Cascading Destruction',
  description:
    'Row of three walls along the flow path. Hydrostatic load builds, the upstream wall fails first, the downstream walls inherit a new surge.',
  setup(ctx) {
    const g = ctx.solver.grid;
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x666666 });
    const chunks: ChunkEntry[] = [];
    const positions = [0.3, 0.5, 0.7].map((f) => Math.floor(g.width * f));
    for (const px of positions) {
      const region = { x: px, y: 0, w: 1, h: g.height };
      const wall = new Float32Array(g.height).fill(3);
      ctx.solver.writeBedRegion(region, wall);
      const mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(g.dx, 3, g.height * g.dx), wallMat.clone()));
      mesh.position.set(g.origin[0] + (px + 0.5) * g.dx, 1.5, g.origin[1] + (g.height * g.dx) / 2);
      ctx.scene.add(mesh);
      chunks.push({ cells: region, topY: 3, mesh, stress: createStress(60, 0.92) });
    }
    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.chunks = chunks;
    ctx.scratch.t = 0;
  },
  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    const g = ctx.solver.grid;
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    // Constant inflow at west edge.
    const tide = Math.min(3.5, 0.4 * (ctx.scratch.t as number));
    const data = new Float32Array(g.height * 2);
    for (let j = 0; j < g.height; j++) { data[j * 2] = tide; data[j * 2 + 1] = tide; }
    ctx.solver.ctx.queue.writeTexture(
      { texture: ctx.solver.waterTex, origin: { x: 0, y: 0 } },
      data,
      { bytesPerRow: 8, rowsPerImage: g.height },
      { width: 1, height: g.height, depthOrArrayLayers: 1 },
    );
    // For each surviving chunk, approximate hydrostatic load = ½ρgh²·L using tide-derived h.
    const chunks = ctx.scratch.chunks as ChunkEntry[];
    for (const c of chunks) {
      if (!c.mesh.visible) continue;
      // Coarse approximation: just use current tide depth as h.
      const F = 0.5 * 1000 * 9.81 * tide * tide * g.height * g.dx;
      if (tickStress(c.stress, F, dt)) {
        // Drop the wall.
        c.mesh.visible = false;
        const blank = new Float32Array(c.cells.h).fill(0);
        ctx.solver.writeBedRegion(c.cells, blank);
      }
    }
  },
};

export default demo;
