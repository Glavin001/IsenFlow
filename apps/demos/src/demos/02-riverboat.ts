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
    const initWater = new Float32Array(g.width * g.height).fill(0.6);
    ctx.solver.writeWaterFull(initWater);

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
    ctx.solver.writeWaterRegion({ x: 0, y: 0, w: 1, h: g.height }, inflow);
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
