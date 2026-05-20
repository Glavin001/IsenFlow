import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '02-riverboat',
  label: 'B. Riverboat',
  description:
    'A river flows along the X axis with constant inflow at the west edge. A wooden crate floats and is carried downstream by hydrodynamic drag (real water→body coupling).',
  setup(ctx) {
    const g = ctx.solver.grid;
    // Pre-fill the channel uniformly with shallow water.
    const initWater = new Float32Array(g.width * g.height).fill(0.6);
    ctx.solver.writeWaterFull(initWater);

    // Banks: raise bed at top/bottom rows. Use ~12 % of grid height each
    // side so the channel scales with whatever world size we configure.
    const bankCells = Math.max(8, Math.round(g.height * 0.12));
    const bank = new Float32Array(g.width).fill(2);
    for (let j = 0; j < bankCells; j++) ctx.solver.writeBedRegion({ x: 0, y: j, w: g.width, h: 1 }, bank);
    for (let j = g.height - bankCells; j < g.height; j++) ctx.solver.writeBedRegion({ x: 0, y: j, w: g.width, h: 1 }, bank);

    // Crate: 0.4m × 0.2m × 0.4m, density of pine wood (~400 kg/m³).
    const halfX = 0.2, halfY = 0.1, halfZ = 0.2;
    const crateGeo = new THREE.BoxGeometry(halfX * 2, halfY * 2, halfZ * 2);
    const crateMat = new THREE.MeshStandardMaterial({ color: 0x9b6a3a });
    const crate = ownByDemo(new THREE.Mesh(crateGeo, crateMat));
    crate.name = 'crate';
    const startX = g.origin[0] + g.width * g.dx * 0.2;
    crate.position.set(startX, 0.3, 0);
    ctx.scene.add(crate);

    const body = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.dynamic()
        .setTranslation(startX, 0.3, 0)
        .setLinearDamping(0.05)
        .setAngularDamping(0.5),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.cuboid(halfX, halfY, halfZ).setDensity(400),
      body,
    );

    ctx.spawnCoupledBody({
      name: 'crate',
      body,
      halfExtents: [halfX, halfY, halfZ],
      mesh: crate,
    });

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
  },
  tick(ctx) {
    (ctx.scratch.water as WaterSurface).update();
    // Re-inject inflow each frame at west edge to keep the river flowing.
    const g = ctx.solver.grid;
    const inflow = new Float32Array(g.height).fill(1.2);
    ctx.solver.writeWaterRegion({ x: 0, y: 0, w: 1, h: g.height }, inflow);
    // No manual addForce — the crate is now driven by real GPU-computed
    // drag and hydrostatic forces accumulated from the per-cell water field.
  },
};

export default demo;
