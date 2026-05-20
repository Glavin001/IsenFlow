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
    // Seed the water texture directly via writeWaterCell ×N — costly but simple.
    // Faster: encode in one writeTexture call:
    const data = new Float32Array(w * h * 2);
    for (let i = 0; i < init.length; i++) { data[i * 2] = init[i]!; data[i * 2 + 1] = init[i]!; }
    ctx.solver.ctx.queue.writeTexture(
      { texture: ctx.solver.waterTex },
      data,
      { bytesPerRow: w * 8, rowsPerImage: h },
      { width: w, height: h, depthOrArrayLayers: 1 },
    );
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
    damMesh.position.set(g.origin[0] + (fillEnd + 0.5) * g.dx, 3, g.origin[1] + (h * g.dx) / 2);
    ctx.scene.add(damMesh);

    // Tiny "village": three boxes downstream.
    const houseMat = new THREE.MeshStandardMaterial({ color: 0xc7a36a, roughness: 0.8 });
    for (let k = 0; k < 3; k++) {
      const hm = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(3, 2.5, 3), houseMat));
      hm.position.set(8 + k * 6, 1.25, -8 + k * 6);
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
