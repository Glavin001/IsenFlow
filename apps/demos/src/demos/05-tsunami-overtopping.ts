import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '05-tsunami',
  label: 'E. Tsunami Overtopping',
  description:
    'A constant inflow at the west edge builds a wave that crests a sea wall and floods the low-lying area behind it.',
  setup(ctx) {
    const g = ctx.solver.grid;
    // Sea wall halfway across.
    const wallX = Math.floor(g.width / 2);
    const wall = new Float32Array(g.height).fill(2.2);
    ctx.solver.writeBedRegion({ x: wallX, y: 0, w: 1, h: g.height }, wall);

    const wallMesh = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(g.dx, 2.2, g.height * g.dx),
      new THREE.MeshStandardMaterial({ color: 0x445566 }),
    ));
    wallMesh.position.set(g.origin[0] + (wallX + 0.5) * g.dx, 1.1, g.origin[1] + (g.height * g.dx) / 2);
    ctx.scene.add(wallMesh);

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.t = 0;
  },
  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    const g = ctx.solver.grid;
    // Increasing inflow at west edge.
    const tideH = Math.min(3.5, 0.5 * (ctx.scratch.t as number));
    const data = new Float32Array(g.height * 2);
    for (let j = 0; j < g.height; j++) { data[j * 2] = tideH; data[j * 2 + 1] = tideH; }
    ctx.solver.ctx.queue.writeTexture(
      { texture: ctx.solver.waterTex, origin: { x: 0, y: 0 } },
      data,
      { bytesPerRow: 8, rowsPerImage: g.height },
      { width: 1, height: g.height, depthOrArrayLayers: 1 },
    );
  },
};

export default demo;
