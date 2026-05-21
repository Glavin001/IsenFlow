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

    // Solid pond floor (static collider) so the concrete block has somewhere
    // to settle. Without this, bodies would fall through the world the moment
    // buoyancy can no longer support them.
    const floorBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.fixed().setTranslation(0, -0.05, 0),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.cuboid(worldW / 2, 0.05, worldW / 2),
      floorBody,
    );

    const bridge = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(worldW * 0.85, 0.15, worldW * 0.18),
      new THREE.MeshStandardMaterial({ color: 0x555555 }),
    ));
    bridge.name = 'bridge';
    bridge.position.set(0, 18, 0);
    ctx.scene.add(bridge);

    // Wooden block (light): density 400 kg/m³.
    const halfSide = 0.25;
    const wood = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(halfSide * 2, halfSide * 2, halfSide * 2),
      new THREE.MeshStandardMaterial({ color: 0xaa6633 }),
    ));
    wood.name = 'wood';
    const woodX = -worldW * 0.18;
    // Drop from well above the bridge so the spec's `y > 10` pre-condition
    // remains true even when several Rapier ticks have run by the time the
    // test reads the body's position (openDemo waits for tickCount > 0; under
    // heavy parallel load that can be several frames of free-fall).
    const dropY = 20;
    wood.position.set(woodX, dropY, 0);
    ctx.scene.add(wood);
    const woodBody = ctx.world.createRigidBody(
      // High linearDamping models water drag and stabilises the body against
      // the GPU coupling's ~3-frame readback latency. Lock pitch & roll so
      // the block stays upright through impact and water-coupled forces.
      ctx.rapier.RigidBodyDesc.dynamic()
        .setTranslation(woodX, dropY, 0)
        .setLinearDamping(3.0)
        .setAngularDamping(5.0)
        .enabledRotations(false, true, false),
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
      waterLevelRef: 1.5, // pond depth
      bedLevelRef: 0,
    });

    // Concrete block (heavy): density 2400 kg/m³ — > 2× water, sinks.
    const concrete = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(halfSide * 2, halfSide * 2, halfSide * 2),
      new THREE.MeshStandardMaterial({ color: 0x888888 }),
    ));
    concrete.name = 'concrete';
    const concX = worldW * 0.18;
    concrete.position.set(concX, dropY, 0);
    ctx.scene.add(concrete);
    const concBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.dynamic()
        .setTranslation(concX, dropY, 0)
        .setLinearDamping(3.0)
        .setAngularDamping(5.0)
        .enabledRotations(false, true, false),
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
      waterLevelRef: 1.5,
      bedLevelRef: 0,
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
