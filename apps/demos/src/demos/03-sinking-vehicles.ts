import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '03-sinking-vehicles',
  label: 'C. Sinking Vehicles',
  description:
    'Two boxes fall from a bridge into a deep pond — a light wooden one floats, a heavy concrete one sinks slowly.',
  setup(ctx) {
    const g = ctx.solver.grid;
    // Deep pond: 3m water everywhere.
    ctx.solver.writeWaterFull(new Float32Array(g.width * g.height).fill(3));
    // Bridge.
    const bridge = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(40, 0.6, 6),
      new THREE.MeshStandardMaterial({ color: 0x555555 }),
    ));
    bridge.position.set(0, 10, 0);
    ctx.scene.add(bridge);

    // Wooden + concrete blocks.
    const wood = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(1.5, 1.5, 1.5),
      new THREE.MeshStandardMaterial({ color: 0xaa6633 }),
    ));
    wood.position.set(-6, 12, 0);
    ctx.scene.add(wood);
    const concrete = ownByDemo(new THREE.Mesh(
      new THREE.BoxGeometry(1.5, 1.5, 1.5),
      new THREE.MeshStandardMaterial({ color: 0x888888 }),
    ));
    concrete.position.set(6, 12, 0);
    ctx.scene.add(concrete);

    const woodBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.dynamic().setTranslation(-6, 12, 0).setLinearDamping(0.8),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.cuboid(0.75, 0.75, 0.75).setDensity(400),
      woodBody,
    );
    const concBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.dynamic().setTranslation(6, 12, 0).setLinearDamping(0.6),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.cuboid(0.75, 0.75, 0.75).setDensity(2400),
      concBody,
    );

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));

    ctx.scratch.water = water;
    ctx.scratch.wood = wood;
    ctx.scratch.concrete = concrete;
    ctx.scratch.woodBody = woodBody;
    ctx.scratch.concBody = concBody;
  },
  tick(ctx) {
    (ctx.scratch.water as WaterSurface).update();
    const sync = (mesh: THREE.Mesh, body: ReturnType<typeof ctx.world.createRigidBody>, isLight: boolean) => {
      const t = body.translation();
      // crude buoyancy + drag impulse if under "water surface" (y < 3).
      if (t.y < 3) {
        const sub = Math.min(1, (3 - t.y) / 1.5);
        const buoy = isLight ? 12000 : 18000;
        body.addForce({ x: 0, y: buoy * sub, z: 0 }, true);
        const v = body.linvel();
        body.addForce({ x: -v.x * 500, y: -v.y * 800, z: -v.z * 500 }, true);
      }
      mesh.position.set(t.x, t.y, t.z);
      const q = body.rotation();
      mesh.quaternion.set(q.x, q.y, q.z, q.w);
    };
    sync(ctx.scratch.wood as THREE.Mesh, ctx.scratch.woodBody as ReturnType<typeof ctx.world.createRigidBody>, true);
    sync(ctx.scratch.concrete as THREE.Mesh, ctx.scratch.concBody as ReturnType<typeof ctx.world.createRigidBody>, false);
  },
};

export default demo;
