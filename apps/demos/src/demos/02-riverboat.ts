import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '02-riverboat',
  label: 'B. Riverboat',
  description:
    'A river flows along the X axis with constant inflow at the west edge. A wooden crate floats and is carried downstream.',
  setup(ctx) {
    const g = ctx.solver.grid;
    // Pre-fill the channel uniformly with shallow water.
    const data = new Float32Array(g.width * g.height * 2);
    for (let i = 0; i < g.width * g.height; i++) { data[i * 2] = 0.6; data[i * 2 + 1] = 0.6; }
    ctx.solver.ctx.queue.writeTexture(
      { texture: ctx.solver.waterTex }, data,
      { bytesPerRow: g.width * 8, rowsPerImage: g.height },
      { width: g.width, height: g.height, depthOrArrayLayers: 1 },
    );

    // Banks: raise bed at top/bottom rows.
    const bank = new Float32Array(g.width).fill(2);
    for (let j = 0; j < 8; j++) ctx.solver.writeBedRegion({ x: 0, y: j, w: g.width, h: 1 }, bank);
    for (let j = g.height - 8; j < g.height; j++) ctx.solver.writeBedRegion({ x: 0, y: j, w: g.width, h: 1 }, bank);

    // Crate.
    const crateGeo = new THREE.BoxGeometry(2, 1, 2);
    const crateMat = new THREE.MeshStandardMaterial({ color: 0x9b6a3a });
    const crate = ownByDemo(new THREE.Mesh(crateGeo, crateMat));
    crate.position.set(-20, 1.5, 0);
    ctx.scene.add(crate);

    const body = ctx.world.createRigidBody(ctx.rapier.RigidBodyDesc.dynamic().setTranslation(-20, 1.5, 0));
    ctx.world.createCollider(ctx.rapier.ColliderDesc.cuboid(1, 0.5, 1), body);
    ctx.bodies.set(1, body);

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.crate = crate;
    ctx.scratch.body = body;
  },
  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    // Re-inject inflow each frame at west edge.
    const g = ctx.solver.grid;
    const inflow = new Float32Array(g.height).fill(1.2);
    ctx.solver.writeBedRegion({ x: 0, y: 0, w: 1, h: g.height }, new Float32Array(g.height));
    // Use the water texture write directly to set depth at column 0.
    const data = new Float32Array(g.height * 2);
    for (let j = 0; j < g.height; j++) { data[j * 2] = inflow[j]!; data[j * 2 + 1] = inflow[j]!; }
    ctx.solver.ctx.queue.writeTexture(
      { texture: ctx.solver.waterTex, origin: { x: 0, y: 0 } },
      data,
      { bytesPerRow: 8, rowsPerImage: g.height },
      { width: 1, height: g.height, depthOrArrayLayers: 1 },
    );
    // Push the crate manually with a small constant force (simulating drag).
    const body = ctx.scratch.body as ReturnType<typeof ctx.world.createRigidBody>;
    body.addForce({ x: 1500, y: 0, z: 0 }, true);
    const t = body.translation();
    const crate = ctx.scratch.crate as THREE.Mesh;
    crate.position.set(t.x, t.y, t.z);
    const q = body.rotation();
    crate.quaternion.set(q.x, q.y, q.z, q.w);
  },
};

export default demo;
