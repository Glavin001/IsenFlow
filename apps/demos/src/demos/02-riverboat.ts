import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { BoundaryType } from 'isenflow';

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

    // ~0.3% bed slope east-to-west so Manning friction + head gradient
    // produce a steady current ≈ 0.5 m/s mid-channel.
    const slope = 0.003;
    const bedSlope = new Float32Array(g.width * g.height);
    for (let j = bankCells; j < g.height - bankCells; j++) {
      for (let i = 0; i < g.width; i++) {
        bedSlope[j * g.width + i] = slope * (g.width - i) * g.dx;
      }
    }
    ctx.solver.writeBedRegion({ x: 0, y: bankCells, w: g.width, h: g.height - 2 * bankCells }, bedSlope.subarray(bankCells * g.width, (g.height - bankCells) * g.width));

    // Boundary: Inflow on west edge, Open on east edge
    ctx.solver.writeBoundaryRegionTarget(
      { x: 0, y: 0, w: 1, h: g.height },
      BoundaryType.Inflow,
      1.2,
    );
    ctx.solver.writeBoundaryRegionTarget(
      { x: g.width - 1, y: 0, w: 1, h: g.height },
      BoundaryType.Open,
      0,
    );

    // Solid river bed: a wide static collider at y=0 so dynamic bodies cannot
    // tunnel through if the GPU buoyancy coupling momentarily misbehaves.
    const worldW = g.width * g.dx;
    const bed = ctx.world.createRigidBody(ctx.rapier.RigidBodyDesc.fixed().setTranslation(0, -0.05, 0));
    ctx.world.createCollider(ctx.rapier.ColliderDesc.cuboid(worldW / 2, 0.05, worldW / 2), bed);

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
        .setLinearDamping(2.0)
        .setAngularDamping(1.0),
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
    // Inflow/Open boundaries handle the river flow via boundaryTargetH —
    // no per-frame writeWaterRegion needed.
  },
};

export default demo;
