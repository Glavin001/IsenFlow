import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { BoundaryType } from 'isenflow';

const WATER_DEPTH = 0.8;

const demo: Demo = {
  id: '02-riverboat',
  label: 'B. Riverboat',
  description:
    'A river flows along the X axis with constant inflow at the west edge. A wooden crate floats and is carried downstream by hydrodynamic drag (real water→body coupling).',
  setup(ctx) {
    const g = ctx.solver.grid;
    const bankCells = Math.max(8, Math.round(g.height * 0.12));

    // Flat bed (no slope) — flow is driven purely by boundary head difference.
    // Fill the channel with water; banks stay dry (bed=2m >> water depth).
    const initWater = new Float32Array(g.width * g.height);
    for (let j = bankCells; j < g.height - bankCells; j++) {
      for (let i = 0; i < g.width; i++) {
        initWater[j * g.width + i] = WATER_DEPTH;
      }
    }
    ctx.solver.writeWaterFull(initWater);

    // Banks: raise bed at top/bottom rows.
    const bank = new Float32Array(g.width).fill(2);
    for (let j = 0; j < bankCells; j++) ctx.solver.writeBedRegion({ x: 0, y: j, w: g.width, h: 1 }, bank);
    for (let j = g.height - bankCells; j < g.height; j++) ctx.solver.writeBedRegion({ x: 0, y: j, w: g.width, h: 1 }, bank);

    // Boundary: Sea at both edges with a small head difference drives steady flow.
    // Open boundary doesn't truly let water exit (out-of-bounds bed=1e6),
    // so we use Sea at a lower level instead.
    ctx.solver.writeBoundaryRegionTarget(
      { x: 0, y: bankCells, w: 1, h: g.height - 2 * bankCells },
      BoundaryType.Sea,
      WATER_DEPTH + 0.02,
    );
    ctx.solver.writeBoundaryRegionTarget(
      { x: g.width - 1, y: bankCells, w: 1, h: g.height - 2 * bankCells },
      BoundaryType.Sea,
      WATER_DEPTH - 0.02,
    );

    // Solid river bed collider so dynamic bodies cannot tunnel through.
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
    const spawnY = WATER_DEPTH + 0.1;
    crate.position.set(startX, spawnY, 0);
    ctx.scene.add(crate);

    const body = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.dynamic()
        .setTranslation(startX, spawnY, 0)
        .setLinearDamping(2.0)
        .setAngularDamping(5.0),
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
      waterLevelRef: WATER_DEPTH,
      bedLevelRef: 0,
    });

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
  },
  tick(ctx) {
    (ctx.scratch.water as WaterSurface).update();
  },
};

export default demo;
