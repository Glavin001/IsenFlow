import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '07-splash',
  label: 'G. Splash Showcase',
  description:
    'Periodic stones dropped into a still pond. Each impact triggers the splash particle pool (decorative, decoupled from SWE).',
  setup(ctx) {
    const g = ctx.solver.grid;
    // Still pond, 2m deep everywhere.
    const data = new Float32Array(g.width * g.height * 2);
    for (let i = 0; i < g.width * g.height; i++) { data[i * 2] = 2; data[i * 2 + 1] = 2; }
    ctx.solver.ctx.queue.writeTexture(
      { texture: ctx.solver.waterTex }, data,
      { bytesPerRow: g.width * 8, rowsPerImage: g.height },
      { width: g.width, height: g.height, depthOrArrayLayers: 1 },
    );

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.t = 0;
    ctx.scratch.nextStone = 0.5;

    // Particle visualization: instanced spheres.
    const geom = new THREE.SphereGeometry(0.05, 4, 4);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const im = ownByDemo(new THREE.InstancedMesh(geom, mat, ctx.splashes.capacity));
    im.frustumCulled = false;
    ctx.scene.add(im);
    ctx.scratch.particles = im;
  },
  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    if ((ctx.scratch.t as number) > (ctx.scratch.nextStone as number)) {
      ctx.scratch.nextStone = (ctx.scratch.t as number) + 0.6;
      const px = (Math.random() - 0.5) * 30;
      const pz = (Math.random() - 0.5) * 30;
      ctx.splashes.spawn({ position: [px, 2, pz], intensity: 1.0, upwardSpeed: 6 });
    }
    // Update instanced mesh transforms.
    const im = ctx.scratch.particles as THREE.InstancedMesh;
    const m = new THREE.Matrix4();
    let idx = 0;
    for (const p of ctx.splashes.pool) {
      if (!p.active) continue;
      m.makeTranslation(p.px, p.py, p.pz);
      im.setMatrixAt(idx++, m);
    }
    im.count = idx;
    im.instanceMatrix.needsUpdate = true;
  },
};

export default demo;
