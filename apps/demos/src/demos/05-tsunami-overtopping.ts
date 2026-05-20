import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '05-tsunami',
  label: 'E. Tsunami Overtopping',
  description:
    'A constant inflow at the west edge builds a wave that crests a 2.2 m sea wall and floods the low-lying area behind it.',
  setup(ctx) {
    const g = ctx.solver.grid;
    const wallX = Math.floor(g.width / 2);
    const wall = new Float32Array(g.height).fill(2.2);
    ctx.solver.writeBedRegion({ x: wallX, y: 0, w: 1, h: g.height }, wall);

    const wallMesh = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(g.dx, 2.2, g.height * g.dx),
      new THREE.MeshStandardMaterial({ color: 0x445566 }),
    ));
    wallMesh.name = 'seawall';
    wallMesh.position.set(g.origin[0] + (wallX + 0.5) * g.dx, 1.1, g.origin[1] + (g.height * g.dx) / 2);
    ctx.scene.add(wallMesh);

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.t = 0;
    ctx.scratch.wallX = wallX;
    ctx.scratch.tideH = 0;
  },
  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    const g = ctx.solver.grid;
    const tideH = Math.min(3.5, 0.5 * (ctx.scratch.t as number));
    ctx.scratch.tideH = tideH;
    ctx.solver.writeWaterRegion({ x: 0, y: 0, w: 1, h: g.height }, new Float32Array(g.height).fill(tideH));
  },
};

export default demo;
