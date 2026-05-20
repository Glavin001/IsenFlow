import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';

const demo: Demo = {
  id: '04-building-flood',
  label: 'D. Building Flood',
  description:
    'A small house with a single door fills with water as the level rises outside (Unified Heightfield in action).',
  setup(ctx) {
    const g = ctx.solver.grid;

    // Building footprint: 16×16 cells centered, walls at top=3m, door 4 cells wide.
    const cx = Math.floor(g.width / 2);
    const cz = Math.floor(g.height / 2);
    const halfW = 8;
    // Floor + 4 walls.
    const wallH = 3;
    // North wall
    const N = new Float32Array(2 * halfW + 1).fill(wallH);
    ctx.solver.writeBedRegion({ x: cx - halfW, y: cz - halfW, w: 2 * halfW + 1, h: 1 }, N);
    // South wall (with door in middle 4 cells)
    const S = new Float32Array(2 * halfW + 1).fill(wallH);
    for (let i = halfW - 1; i <= halfW + 2; i++) S[i] = 0; // door sill height 0
    ctx.solver.writeBedRegion({ x: cx - halfW, y: cz + halfW, w: 2 * halfW + 1, h: 1 }, S);
    // East / West walls
    const E = new Float32Array(2 * halfW + 1).fill(wallH);
    ctx.solver.writeBedRegion({ x: cx + halfW, y: cz - halfW, w: 1, h: 2 * halfW + 1 }, E);
    ctx.solver.writeBedRegion({ x: cx - halfW, y: cz - halfW, w: 1, h: 2 * halfW + 1 }, E);

    // Visual.
    const wallMat = new THREE.MeshStandardMaterial({ color: 0xc0a070 });
    const N_mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry((2 * halfW + 1) * g.dx, wallH, g.dx), wallMat));
    N_mesh.position.set(g.origin[0] + cx * g.dx, wallH / 2, g.origin[1] + (cz - halfW + 0.5) * g.dx);
    ctx.scene.add(N_mesh);
    const E_mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(g.dx, wallH, (2 * halfW + 1) * g.dx), wallMat));
    E_mesh.position.set(g.origin[0] + (cx + halfW + 0.5) * g.dx, wallH / 2, g.origin[1] + cz * g.dx);
    ctx.scene.add(E_mesh);
    const W_mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(g.dx, wallH, (2 * halfW + 1) * g.dx), wallMat));
    W_mesh.position.set(g.origin[0] + (cx - halfW + 0.5) * g.dx, wallH / 2, g.origin[1] + cz * g.dx);
    ctx.scene.add(W_mesh);
    // South wall split around door.
    const sLen = halfW - 2;
    const Sl = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(sLen * g.dx, wallH, g.dx), wallMat));
    Sl.position.set(g.origin[0] + (cx - halfW + sLen / 2) * g.dx, wallH / 2, g.origin[1] + (cz + halfW + 0.5) * g.dx);
    ctx.scene.add(Sl);
    const Sr = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(sLen * g.dx, wallH, g.dx), wallMat));
    Sr.position.set(g.origin[0] + (cx + halfW - sLen / 2 + 1) * g.dx, wallH / 2, g.origin[1] + (cz + halfW + 0.5) * g.dx);
    ctx.scene.add(Sr);

    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));
    ctx.scratch.water = water;
    ctx.scratch.t = 0;
  },
  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    // Rising-tide source at the south edge (j = max).
    const g = ctx.solver.grid;
    const tideH = Math.min(2.5, 0.2 * (ctx.scratch.t as number));
    const tideRow = new Float32Array(g.width).fill(tideH);
    ctx.solver.writeWaterRegion({ x: 0, y: g.height - 1, w: g.width, h: 1 }, tideRow);
  },
};

export default demo;
