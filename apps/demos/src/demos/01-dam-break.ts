import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '01-dam-break',
  label: 'A. Classic Dam Break',
  description:
    'Reservoir behind a 4 m tall dam; the dam vanishes at t=2s and the wave rolls across a dry bed toward a small village.',

  setup(ctx: DemoContext) {
    const g = ctx.solver.grid;
    // Reservoir: fill the left third with 4 m of water.
    const w = g.width;
    const h = g.height;
    const fillEnd = Math.floor(w / 3);
    const init = new Float32Array(w * h);
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < fillEnd; i++) init[j * w + i] = 4;
    }
    ctx.solver.writeWaterFull(init);
    // Build the dam by raising the bed to 6 m for one column near fillEnd.
    const dam = new Float32Array(h).fill(6);
    ctx.solver.writeBedRegion({ x: fillEnd, y: 0, w: 1, h }, dam);

    // Visualize dam.
    const damMesh = ownByDemo(
      new THREE.Mesh(
        new THREE.BoxGeometry(g.dx, 6, h * g.dx),
        new THREE.MeshStandardMaterial({ color: 0x806a55, roughness: 0.9 }),
      ),
    );
    damMesh.name = 'dam';
    damMesh.position.set(g.origin[0] + (fillEnd + 0.5) * g.dx, 3, g.origin[1] + (h * g.dx) / 2);
    ctx.scene.add(damMesh);

    const houseMat = new THREE.MeshStandardMaterial({ color: 0xc7a36a, roughness: 0.8 });
    // Three small houses east of the dam, scaled to the world size.
    const worldHalf = (g.width * g.dx) / 2;
    for (let k = 0; k < 3; k++) {
      const hm = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.7, 0.8), houseMat));
      hm.name = `house${k}`;
      hm.position.set(worldHalf * 0.25 + k * worldHalf * 0.2, 0.35, -worldHalf * 0.3 + k * worldHalf * 0.25);
      ctx.scene.add(hm);
    }

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.damMesh = damMesh;
    ctx.scratch.dropAt = 2;
    ctx.scratch.t = 0;
    ctx.scratch.dropped = false;
    ctx.scratch.fillEnd = fillEnd;
  },

  tick(ctx: DemoContext, dt: number) {
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    const w = ctx.scratch.water as WaterSurface;
    w.update();
    if (!ctx.scratch.dropped && (ctx.scratch.t as number) > (ctx.scratch.dropAt as number)) {
      ctx.scratch.dropped = true;
      // Lower the dam column.
      const g = ctx.solver.grid;
      const fillEnd = ctx.scratch.fillEnd as number;
      const dam = new Float32Array(g.height).fill(0);
      ctx.solver.writeBedRegion({ x: fillEnd, y: 0, w: 1, h: g.height }, dam);
      const dm = ctx.scratch.damMesh as THREE.Mesh;
      dm.visible = false;
    }
  },
};

export default demo;
