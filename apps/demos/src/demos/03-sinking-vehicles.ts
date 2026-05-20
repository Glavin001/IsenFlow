import * as THREE from 'three';
import type { Demo } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '03-sinking-vehicles',
  label: 'C. Sinking Vehicles',
  description:
    'Two boxes fall from a bridge into a deep pond. Buoyancy and vertical damping are computed by the GPU water-coupling kernel; wood floats, concrete sinks.',
  setup(ctx) {
    const g = ctx.solver.grid;
    const worldW = g.width * g.dx;
    // Pond depth: 1.5 m everywhere.
    ctx.solver.writeWaterFull(new Float32Array(g.width * g.height).fill(1.5));

    const bridge = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(worldW * 0.85, 0.15, worldW * 0.18),
      new THREE.MeshStandardMaterial({ color: 0x555555 }),
    ));
    bridge.name = 'bridge';
    bridge.position.set(0, 4, 0);
    ctx.scene.add(bridge);

    // Wooden block (light): density 400 kg/m³.
    const halfSide = 0.25;
    const wood = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(halfSide * 2, halfSide * 2, halfSide * 2),
      new THREE.MeshStandardMaterial({ color: 0xaa6633 }),
    ));
    wood.name = 'wood';
    const woodX = -worldW * 0.18;
    wood.position.set(woodX, 5, 0);
    ctx.scene.add(wood);
    const woodBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.dynamic().setTranslation(woodX, 5, 0).setLinearDamping(0.4),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.cuboid(halfSide, halfSide, halfSide).setDensity(400),
      woodBody,
    );
    ctx.spawnCoupledBody({
      name: 'wood',
      body: woodBody,
      halfExtents: [halfSide, halfSide, halfSide],
      mesh: wood,
    });

    // Concrete block (heavy): density 2400 kg/m³ — > 2× water, sinks.
    const concrete = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(halfSide * 2, halfSide * 2, halfSide * 2),
      new THREE.MeshStandardMaterial({ color: 0x888888 }),
    ));
    concrete.name = 'concrete';
    const concX = worldW * 0.18;
    concrete.position.set(concX, 5, 0);
    ctx.scene.add(concrete);
    const concBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.dynamic().setTranslation(concX, 5, 0).setLinearDamping(0.4),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.cuboid(halfSide, halfSide, halfSide).setDensity(2400),
      concBody,
    );
    ctx.spawnCoupledBody({
      name: 'concrete',
      body: concBody,
      halfExtents: [halfSide, halfSide, halfSide],
      mesh: concrete,
    });

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
  },
  tick(ctx) {
    (ctx.scratch.water as WaterSurface).update();
    // No manual buoyancy — coupling kernel handles it.
  },
};

export default demo;
