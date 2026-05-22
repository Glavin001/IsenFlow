import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { BoundaryType, applyImpact, spawnImpactSplash } from 'isenflow';

/**
 * Rain Storm — hundreds of raindrops hit a shallow basin, each creating
 * solver-driven ripples that propagate and interact. Weather cycles from
 * drizzle to torrential storm every ~45 seconds.
 *
 * Visual layers:
 * 1. Instanced rain streaks (falling cylinders)
 * 2. SWE ripples via bed-piston technique
 * 3. Instanced splash particles at impact points
 */

// ── Seeded PRNG ──────────────────────────────────────────────────────

function makeRng(seed: number): () => number {
  let s = seed | 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}

// ── Constants ────────────────────────────────────────────────────────

// WebGPU uniform buffer limit is 65536 bytes; each instance matrix = 64 bytes.
// Three.js packs extra uniforms too, so stay well under 1024 instances.
const RAIN_POOL = 900;        // CPU pool (not all rendered)
const RAIN_VISUAL_CAP = 900;  // max instances in the InstancedMesh
const SPLASH_VISUAL_CAP = 800;
const MAX_PISTONS_PER_FRAME = 30;
const PISTON_RADIUS = 2;
const PISTON_DURATION = 0.08; // seconds
const RAIN_DURATION = 30;  // seconds of rain: 0 → max → 0
const CALM_DURATION = 5;   // seconds of silence between storms
const WEATHER_CYCLE = RAIN_DURATION + CALM_DURATION; // 35s total
const BASE_WATER_DEPTH = 0.15;
const BASIN_WALL_HEIGHT = 0.5;
const SPAWN_Y_MIN = 4;
const SPAWN_Y_MAX = 7;
const FALL_SPEED_MIN = 6;
const FALL_SPEED_MAX = 10;

// ── Weather ──────────────────────────────────────────────────────────

function weatherIntensity(t: number): number {
  // First cycle starts as if already past the calm period (offset by CALM_DURATION).
  const adjusted = t + CALM_DURATION;
  const phase = adjusted % WEATHER_CYCLE;
  if (phase >= RAIN_DURATION) return 0; // calm window
  // Sine 0→1→0 over RAIN_DURATION seconds
  return Math.sin((Math.PI * phase) / RAIN_DURATION);
}

// ── Rain particle pool (SoA) ─────────────────────────────────────────

interface RainPool {
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  vz: Float32Array;
  active: Uint8Array;
  cursor: number;
}

function createRainPool(): RainPool {
  return {
    x: new Float32Array(RAIN_POOL),
    y: new Float32Array(RAIN_POOL),
    z: new Float32Array(RAIN_POOL),
    vx: new Float32Array(RAIN_POOL),
    vy: new Float32Array(RAIN_POOL),
    vz: new Float32Array(RAIN_POOL),
    active: new Uint8Array(RAIN_POOL),
    cursor: 0,
  };
}

// ── Pending bed restore ──────────────────────────────────────────────

interface PendingRestore {
  at: number;
  region: { x: number; y: number; w: number; h: number };
  buf: Float32Array;
}

// ── Demo ─────────────────────────────────────────────────────────────

const demo: Demo = {
  id: '11-rain-storm',
  label: 'K. Rain Storm',
  description:
    'Hundreds of raindrops hit a shallow basin, each creating solver-driven ' +
    'ripples. Weather cycles from drizzle to torrential storm every ~45s. ' +
    'Click to drop a boulder splash.',

  setup(ctx: DemoContext) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;
    const dx = g.dx;
    const rng = makeRng(7777);

    // ── 1. Basin terrain ──
    const terrain = new Float32Array(W * H);
    const borderFrac = 0.15;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        // Distance from edge as fraction (0 = edge, 1 = interior)
        const fx = Math.min(i, W - 1 - i) / (W * borderFrac);
        const fz = Math.min(j, H - 1 - j) / (H * borderFrac);
        const edgeDist = Math.min(fx, fz);
        const t = Math.min(1, Math.max(0, edgeDist));
        // Smooth ramp: raised at edges, flat in center
        const ramp = BASIN_WALL_HEIGHT * (1 - t * t);
        terrain[j * W + i] = ramp;
      }
    }
    ctx.solver.writeBedFull(terrain);

    // ── 2. Boundaries: closed on all sides ──
    for (let j = 0; j < H; j++) {
      ctx.solver.writeBoundaryRegionTarget({ x: 0, y: j, w: 1, h: 1 }, BoundaryType.Closed, 0);
      ctx.solver.writeBoundaryRegionTarget({ x: W - 1, y: j, w: 1, h: 1 }, BoundaryType.Closed, 0);
    }
    for (let i = 0; i < W; i++) {
      ctx.solver.writeBoundaryRegionTarget({ x: i, y: 0, w: 1, h: 1 }, BoundaryType.Closed, 0);
      ctx.solver.writeBoundaryRegionTarget({ x: i, y: H - 1, w: 1, h: 1 }, BoundaryType.Closed, 0);
    }

    // ── 3. Initial water ──
    const water0 = new Float32Array(W * H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const bed = terrain[j * W + i];
        water0[j * W + i] = Math.max(0, BASE_WATER_DEPTH - bed * 0.3);
      }
    }
    ctx.solver.writeWaterFull(water0);

    // ── 4. Terrain mesh ──
    const terrainGeom = new THREE.PlaneGeometry(W * dx, H * dx, W, H);
    terrainGeom.rotateX(-Math.PI / 2);
    const pos = terrainGeom.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    const baseColor = new THREE.Color(0x1a2a3a); // dark blue-gray pond floor
    const edgeColor = new THREE.Color(0x2a3a2a); // dark green-gray edges
    for (let idx = 0; idx < pos.count; idx++) {
      const px = pos.getX(idx);
      const pz = pos.getZ(idx);
      // Sample terrain at this vertex
      const ci = Math.round((px - g.origin[0]) / dx);
      const cj = Math.round((pz - g.origin[1]) / dx);
      const ci2 = Math.max(0, Math.min(W - 1, ci));
      const cj2 = Math.max(0, Math.min(H - 1, cj));
      const h = terrain[cj2 * W + ci2];
      pos.setY(idx, h);
      const c = new THREE.Color().lerpColors(baseColor, edgeColor, Math.min(1, h / BASIN_WALL_HEIGHT));
      c.toArray(colors, idx * 3);
    }
    terrainGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    terrainGeom.computeVertexNormals();
    const terrainMesh = ownByDemo(
      new THREE.Mesh(
        terrainGeom,
        new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92 }),
      ),
    );
    terrainMesh.name = 'terrain';
    terrainMesh.receiveShadow = true;
    ctx.scene.add(terrainMesh);

    // ── 5. Water surface ──
    const waterSurf = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(waterSurf.mesh));

    // ── 6. Rain streak instanced meshes (split in two to stay under WebGPU uniform limit) ──
    const HALF = Math.ceil(RAIN_VISUAL_CAP / 2);
    const rainGeom = new THREE.CylinderGeometry(0.006, 0.006, 0.3, 3);
    const rainMat = new THREE.MeshBasicMaterial({
      color: 0xccddff,
      transparent: true,
      opacity: 0.6,
    });
    const rainMeshA = ownByDemo(new THREE.InstancedMesh(rainGeom, rainMat, HALF));
    rainMeshA.name = 'rainStreaksA';
    rainMeshA.frustumCulled = false;
    rainMeshA.count = 0;
    ctx.scene.add(rainMeshA);
    const rainMeshB = ownByDemo(new THREE.InstancedMesh(rainGeom, rainMat, HALF));
    rainMeshB.name = 'rainStreaksB';
    rainMeshB.frustumCulled = false;
    rainMeshB.count = 0;
    ctx.scene.add(rainMeshB);

    // ── 7. Splash instanced mesh ──
    const splashGeom = new THREE.SphereGeometry(0.012, 4, 3);
    const splashMat = new THREE.MeshBasicMaterial({
      color: 0xddeeff,
      transparent: true,
      opacity: 0.5,
    });
    const splashMesh = ownByDemo(new THREE.InstancedMesh(splashGeom, splashMat, SPLASH_VISUAL_CAP));
    splashMesh.name = 'splashDroplets';
    splashMesh.frustumCulled = false;
    splashMesh.count = 0;
    ctx.scene.add(splashMesh);

    // ── 8. Click interaction ──
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const onPointerDown = (event: PointerEvent) => {
      const rect = ctx.renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, ctx.camera);
      const hits = raycaster.intersectObject(terrainMesh);
      if (hits.length > 0) {
        const pt = hits[0].point;
        const result = applyImpact(ctx.solver, {
          worldX: pt.x,
          worldZ: pt.z,
          velocity: 8 + rng() * 6,
          shape: 'sphere',
          size: 0.3 + rng() * 0.4,
          density: 2500,
          waterDepth: BASE_WATER_DEPTH,
        });
        spawnImpactSplash(ctx.splashes, {
          worldX: pt.x,
          worldZ: pt.z,
          velocity: 10,
          shape: 'sphere',
          size: 0.35,
          density: 2500,
          waterDepth: BASE_WATER_DEPTH,
        }, result);
      }
    };
    ctx.renderer.domElement.addEventListener('pointerdown', onPointerDown);

    // ── 9. Fog atmosphere ──
    ctx.scene.fog = new THREE.FogExp2(0x334455, 0.04);

    // ── 10. Camera ──
    ctx.camera.position.set(-6, 8, 10);
    ctx.controls.target.set(0, 0.1, 0);
    ctx.controls.update();

    // ── Scratch state ──
    ctx.scratch.water = waterSurf;
    ctx.scratch.t = 0;
    ctx.scratch.terrain = terrain;
    ctx.scratch.rng = rng;
    ctx.scratch.rainPool = createRainPool();
    ctx.scratch.rainMeshA = rainMeshA;
    ctx.scratch.rainMeshB = rainMeshB;
    ctx.scratch.splashMesh = splashMesh;
    ctx.scratch.spawnAccum = 0;
    ctx.scratch.pendingRestores = [] as PendingRestore[];
    ctx.scratch.onPointerDown = onPointerDown;
  },

  tick(ctx: DemoContext, dt: number) {
    const waterSurf = ctx.scratch.water as WaterSurface;
    waterSurf.update();

    const t = (ctx.scratch.t as number) + dt;
    ctx.scratch.t = t;
    const rng = ctx.scratch.rng as () => number;
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;
    const dx = g.dx;
    const terrain = ctx.scratch.terrain as Float32Array;
    const pool = ctx.scratch.rainPool as RainPool;
    const rainMeshA = ctx.scratch.rainMeshA as THREE.InstancedMesh;
    const rainMeshB = ctx.scratch.rainMeshB as THREE.InstancedMesh;
    const splashMesh = ctx.scratch.splashMesh as THREE.InstancedMesh;

    const weather = weatherIntensity(t);

    // ── Wind ──
    const windAngle = t * 0.3;
    const windSpeed = weather * 2.5;
    const windX = Math.sin(windAngle) * windSpeed;
    const windZ = Math.cos(windAngle) * windSpeed * 0.3;

    // ── Spawn new rain particles (rate defined as drops/second, frame-rate independent) ──
    // Calm: ~150 drops/s → ~100 in air. Peak: ~1500 drops/s → ~750 in air.
    const dropsPerSec = 150 + weather * 1350;
    let spawnAccum = (ctx.scratch.spawnAccum as number) + dropsPerSec * dt;
    const spawnCount = Math.floor(spawnAccum);
    spawnAccum -= spawnCount;
    ctx.scratch.spawnAccum = spawnAccum;
    const domainMin = g.origin[0] + 1;
    const domainMax = g.origin[0] + (W - 2) * dx;
    const domainMinZ = g.origin[1] + 1;
    const domainMaxZ = g.origin[1] + (H - 2) * dx;

    for (let s = 0; s < spawnCount; s++) {
      const idx = pool.cursor;
      pool.cursor = (pool.cursor + 1) % RAIN_POOL;
      pool.active[idx] = 1;
      pool.x[idx] = domainMin + rng() * (domainMax - domainMin);
      pool.y[idx] = SPAWN_Y_MIN + rng() * (SPAWN_Y_MAX - SPAWN_Y_MIN);
      pool.z[idx] = domainMinZ + rng() * (domainMaxZ - domainMinZ);
      pool.vy[idx] = -(FALL_SPEED_MIN + rng() * (FALL_SPEED_MAX - FALL_SPEED_MIN));
      pool.vx[idx] = windX + (rng() - 0.5) * 0.5;
      pool.vz[idx] = windZ + (rng() - 0.5) * 0.5;
    }

    // ── Update rain particles & detect hits ──
    const hits: Array<{ x: number; z: number }> = [];
    const waterY = BASE_WATER_DEPTH; // approximate surface Y

    for (let i = 0; i < RAIN_POOL; i++) {
      if (!pool.active[i]) continue;
      pool.x[i] += pool.vx[i] * dt;
      pool.y[i] += pool.vy[i] * dt;
      pool.z[i] += pool.vz[i] * dt;

      if (pool.y[i] <= waterY) {
        pool.active[i] = 0;
        // Only register hit if within domain
        if (pool.x[i] > domainMin && pool.x[i] < domainMax &&
            pool.z[i] > domainMinZ && pool.z[i] < domainMaxZ) {
          hits.push({ x: pool.x[i], z: pool.z[i] });
        }
      }
    }

    // ── Apply bed pistons (throttled) ──
    const pending = ctx.scratch.pendingRestores as PendingRestore[];
    let pistonBudget = MAX_PISTONS_PER_FRAME;
    const R = PISTON_RADIUS;
    const diam = R * 2 + 1;
    const pulseHeight = 0.05 + weather * 0.20;

    for (const hit of hits) {
      // Spawn splash for every hit
      ctx.splashes.spawn({
        position: [hit.x, waterY, hit.z],
        intensity: 0.3 + weather * 0.5,
        upwardSpeed: 2.5 + weather * 3.5,
      });

      if (pistonBudget <= 0) continue;
      pistonBudget--;

      // Convert to grid coords
      const ci = Math.round((hit.x - g.origin[0]) / dx);
      const cj = Math.round((hit.z - g.origin[1]) / dx);
      if (ci < R + 1 || ci >= W - R - 1 || cj < R + 1 || cj >= H - R - 1) continue;

      // Build cosine-bell piston patch
      const buf = new Float32Array(diam * diam);
      const restoreBuf = new Float32Array(diam * diam);
      for (let dj = 0; dj < diam; dj++) {
        for (let di = 0; di < diam; di++) {
          const dr = Math.hypot(di - R, dj - R) / R;
          const base = terrain[(cj - R + dj) * W + (ci - R + di)] ?? 0;
          restoreBuf[dj * diam + di] = base;
          buf[dj * diam + di] = dr < 1.0
            ? base + pulseHeight * 0.5 * (1 + Math.cos(Math.PI * dr))
            : base;
        }
      }
      const region = { x: ci - R, y: cj - R, w: diam, h: diam };
      ctx.solver.writeBedRegion(region, buf);
      pending.push({ at: t + PISTON_DURATION, region, buf: restoreBuf });
    }

    // ── Process pending bed restores ──
    for (let i = pending.length - 1; i >= 0; i--) {
      if (t >= pending[i].at) {
        ctx.solver.writeBedRegion(pending[i].region, pending[i].buf);
        pending.splice(i, 1);
      }
    }

    // ── Update rain instanced meshes (split across A/B to stay under WebGPU limit) ──
    const HALF = Math.ceil(RAIN_VISUAL_CAP / 2);
    const tempObj = new THREE.Object3D();
    let activeCount = 0;
    for (let i = 0; i < RAIN_POOL; i++) {
      if (!pool.active[i]) continue;
      if (activeCount >= RAIN_VISUAL_CAP) break;
      tempObj.position.set(pool.x[i], pool.y[i], pool.z[i]);
      const speed = Math.sqrt(pool.vx[i] ** 2 + pool.vy[i] ** 2 + pool.vz[i] ** 2);
      const speedFactor = speed / FALL_SPEED_MIN;
      tempObj.scale.set(1, speedFactor, 1);
      tempObj.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(pool.vx[i] / speed, pool.vy[i] / speed, pool.vz[i] / speed),
      );
      tempObj.updateMatrix();
      const mesh = activeCount < HALF ? rainMeshA : rainMeshB;
      const idx = activeCount < HALF ? activeCount : activeCount - HALF;
      mesh.setMatrixAt(idx, tempObj.matrix);
      activeCount++;
    }
    rainMeshA.count = Math.min(activeCount, HALF);
    rainMeshA.instanceMatrix.needsUpdate = true;
    rainMeshB.count = Math.max(0, activeCount - HALF);
    if (rainMeshB.count > 0) rainMeshB.instanceMatrix.needsUpdate = true;

    // ── Update splash instanced mesh ──
    let splashCount = 0;
    for (const p of ctx.splashes.pool) {
      if (!p.active) continue;
      if (splashCount >= SPLASH_VISUAL_CAP) break;
      tempObj.position.set(p.px, p.py, p.pz);
      tempObj.scale.setScalar(1);
      tempObj.quaternion.identity();
      tempObj.updateMatrix();
      splashMesh.setMatrixAt(splashCount, tempObj.matrix);
      splashCount++;
    }
    splashMesh.count = splashCount;
    if (splashCount > 0) splashMesh.instanceMatrix.needsUpdate = true;
  },

  cleanup(ctx: DemoContext) {
    const handler = ctx.scratch.onPointerDown as EventListener;
    if (handler) {
      ctx.renderer.domElement.removeEventListener('pointerdown', handler);
    }
    ctx.scene.fog = null;
  },
};

export default demo;
