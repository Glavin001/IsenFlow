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
    ctx.solver.writeWaterFull(new Float32Array(g.width * g.height).fill(2));

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.t = 0;
    ctx.scratch.nextStone = 0.5;

    // Particle visualization: a single THREE.Points with one position attribute.
    // (Three.js's WebGPU InstancedMesh path packs object uniforms into a single
    //  uniform buffer; for ~4 k instances that crosses the 64 KB WebGPU limit
    //  and the pipeline is invalidated. Points sidesteps that entirely.)
    const cap = ctx.splashes.capacity;
    const positions = new Float32Array(cap * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setDrawRange(0, 0);
    const mat = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.18,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    });
    const points = ownByDemo(new THREE.Points(geom, mat));
    points.frustumCulled = false;
    ctx.scene.add(points);
    ctx.scratch.particlesPositions = positions;
    ctx.scratch.particlesGeom = geom;
    ctx.scratch.particles = points;
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
    const positions = ctx.scratch.particlesPositions as Float32Array;
    const geom = ctx.scratch.particlesGeom as THREE.BufferGeometry;
    let n = 0;
    for (const p of ctx.splashes.pool) {
      if (!p.active) continue;
      positions[n * 3 + 0] = p.px;
      positions[n * 3 + 1] = p.py;
      positions[n * 3 + 2] = p.pz;
      n++;
    }
    geom.setDrawRange(0, n);
    geom.attributes.position!.needsUpdate = true;
    geom.computeBoundingSphere();
  },
};

export default demo;
