import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { BoundaryType, applyImpact, spawnImpactSplash } from 'isenflow';

/**
 * Stress Test — massive terrain with a huge lake, 100+ buildings/obstacles,
 * multiple flood sources, periodic dam breaks, floating debris, and big waves.
 *
 * Designed to push the SWE solver, rasterizer, and renderer to their limits.
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

// ── Terrain generation ───────────────────────────────────────────────

interface Peak {
  cx: number; cz: number;
  amp: number;
  sx: number; sz: number;
}

// Multi-kilometer scale terrain (mapped to 16m grid).
// Imagine 1 grid-meter = 250 real meters → 16m grid = 4 km.
const MOUNTAINS: Peak[] = [
  // Major mountain range (north)
  { cx: -3.0, cz: -6.0, amp: 5.0, sx: 2.0, sz: 1.5 },
  { cx:  0.0, cz: -5.5, amp: 4.5, sx: 1.8, sz: 1.6 },
  { cx:  3.5, cz: -5.0, amp: 4.8, sx: 2.2, sz: 1.4 },
  { cx:  6.0, cz: -4.5, amp: 3.8, sx: 1.5, sz: 1.3 },
  // Ridge connecting peaks
  { cx:  1.5, cz: -5.5, amp: 3.0, sx: 1.2, sz: 0.8 },
  { cx: -1.5, cz: -5.8, amp: 2.8, sx: 1.0, sz: 0.7 },

  // Eastern highlands
  { cx:  6.5, cz: -1.0, amp: 3.5, sx: 1.5, sz: 2.0 },
  { cx:  5.5, cz:  2.0, amp: 2.8, sx: 1.2, sz: 1.5 },

  // Western hills
  { cx: -6.0, cz: -2.0, amp: 3.0, sx: 1.5, sz: 1.8 },
  { cx: -5.5, cz:  1.5, amp: 2.2, sx: 1.3, sz: 1.2 },

  // Southern foothills
  { cx: -3.0, cz:  5.0, amp: 1.8, sx: 1.5, sz: 1.0 },
  { cx:  3.0, cz:  5.5, amp: 1.5, sx: 1.2, sz: 1.0 },
  { cx:  0.0, cz:  6.5, amp: 1.0, sx: 2.0, sz: 0.8 },

  // Lake basin rim (forms natural dam around central lake)
  { cx: -2.5, cz:  0.5, amp: 2.0, sx: 0.8, sz: 1.5 },
  { cx:  2.5, cz:  0.5, amp: 2.0, sx: 0.8, sz: 1.5 },
  { cx:  0.0, cz:  2.5, amp: 1.4, sx: 2.5, sz: 0.5 },

  // Small bumps for terrain texture
  { cx: -4.0, cz:  3.5, amp: 0.6, sx: 0.5, sz: 0.5 },
  { cx:  4.5, cz:  4.0, amp: 0.5, sx: 0.5, sz: 0.5 },
  { cx: -1.0, cz: -3.0, amp: 0.4, sx: 0.4, sz: 0.4 },
  { cx:  2.0, cz: -3.0, amp: 0.5, sx: 0.5, sz: 0.4 },
];

const VALLEYS: Peak[] = [
  // Central lake basin (large depression)
  { cx:  0.0, cz: -0.5, amp: -1.5, sx: 2.5, sz: 2.0 },
  // River channel: north mountains → lake
  { cx:  0.0, cz: -3.5, amp: -0.8, sx: 0.5, sz: 1.5 },
  { cx:  0.0, cz: -1.5, amp: -1.0, sx: 0.6, sz: 1.0 },
  // Gorge: lake overflow → south
  { cx:  0.0, cz:  3.0, amp: -0.6, sx: 0.4, sz: 1.0 },
  { cx:  0.2, cz:  5.0, amp: -0.5, sx: 0.5, sz: 1.5 },
  // Western river channel
  { cx: -4.0, cz: -0.5, amp: -0.5, sx: 0.4, sz: 2.0 },
  // Eastern ravine
  { cx:  5.0, cz:  0.0, amp: -0.4, sx: 0.3, sz: 2.5 },
];

function sampleTerrain(wx: number, wz: number): number {
  let h = 0;
  for (const p of MOUNTAINS) {
    const ddx = wx - p.cx;
    const ddz = wz - p.cz;
    h += p.amp * Math.exp(-(ddx * ddx) / (2 * p.sx * p.sx) - (ddz * ddz) / (2 * p.sz * p.sz));
  }
  for (const p of VALLEYS) {
    const ddx = wx - p.cx;
    const ddz = wz - p.cz;
    h += p.amp * Math.exp(-(ddx * ddx) / (2 * p.sx * p.sx) - (ddz * ddz) / (2 * p.sz * p.sz));
  }
  return Math.max(0, h);
}

// ── Obstacle system ──────────────────────────────────────────────────
//
// Single source of truth: each obstacle is pure data (cell position,
// half-extents, wall height above terrain, color). The pipeline is:
//
//   1. terrain[]       — base heightmap from gaussian peaks
//   2. obstacles[]     — list of Obstacle descriptors (pure data, no side effects)
//   3. compositeBed()  — builds final bed[] = max(terrain, all obstacle tops)
//                         per cell, written to solver ONCE via writeBedFull
//   4. spawnVisuals()  — creates Three.js meshes positioned from the same
//                         composited bed, so physics & visuals never diverge
//

/** Pure data: describes a rectangular obstacle sitting on the terrain. */
interface Obstacle {
  /** Grid cell center X. */
  cx: number;
  /** Grid cell center Z. */
  cz: number;
  /** Half-extent in cells along X. */
  halfW: number;
  /** Half-extent in cells along Z. */
  halfH: number;
  /** Wall height in meters ABOVE the terrain at each cell. */
  wallH: number;
  /** Visual color. */
  color: number;
  /** Unique name for scene graph lookup (e.g. dam break removal). */
  name: string;
}

function generateObstacles(W: number, H: number, dx: number, originX: number, originZ: number): Obstacle[] {
  const obstacles: Obstacle[] = [];
  const rng = makeRng(12345);
  let id = 0;

  const canPlace = (ci: number, cj: number, hw: number, hh: number): boolean => {
    const wx = originX + ci * dx;
    const wz = originZ + cj * dx;
    const h = sampleTerrain(wx, wz);
    if (h > 1.5 || h < 0.05) return false;
    if (ci - hw < 3 || ci + hw >= W - 3 || cj - hh < 3 || cj + hh >= H - 3) return false;
    return true;
  };

  // Downtown towers
  const dtSpacing = Math.round(W * 0.055);
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 8; col++) {
      if ((row + col) % 3 === 0) continue;
      const ci = Math.round(W * 0.15) + col * dtSpacing;
      const cj = Math.round(H * 0.55) + row * Math.round(H * 0.045);
      const hw = Math.round(3 + rng() * 5);
      const hh = Math.round(3 + rng() * 5);
      if (!canPlace(ci, cj, hw, hh)) continue;
      const wallH = 1.0 + rng() * 2.5;
      const lum = Math.round(0x66 + rng() * 0x44);
      obstacles.push({ cx: ci, cz: cj, halfW: hw, halfH: hh, wallH,
        color: (lum << 16) | ((lum - 0x10) << 8) | (lum + 0x10), name: `tower${id++}` });
    }
  }

  // Warehouses
  for (let i = 0; i < 8; i++) {
    const ci = Math.round(W * 0.72 + rng() * W * 0.12);
    const cj = Math.round(W * 0.25 + i * Math.round(H * 0.08));
    const hw = Math.round(5 + rng() * 8);
    const hh = Math.round(3 + rng() * 5);
    if (!canPlace(ci, cj, hw, hh)) continue;
    obstacles.push({ cx: ci, cz: cj, halfW: hw, halfH: hh,
      wallH: 0.8 + rng() * 1.0, color: 0x556655 + Math.round(rng() * 0x111111), name: `warehouse${id++}` });
  }

  // Residential houses
  for (let i = 0; i < 40; i++) {
    const ci = Math.round(10 + rng() * (W - 20));
    const cj = Math.round(10 + rng() * (H - 20));
    const hw = Math.round(2 + rng() * 3);
    const hh = Math.round(2 + rng() * 3);
    if (!canPlace(ci, cj, hw, hh)) continue;
    obstacles.push({ cx: ci, cz: cj, halfW: hw, halfH: hh,
      wallH: 0.5 + rng() * 0.8, color: 0xc0a070 + Math.round(rng() * 0x202020), name: `house${id++}` });
  }

  // Dam
  obstacles.push({ cx: Math.round(W * 0.50), cz: Math.round(H * 0.68),
    halfW: Math.round(W * 0.35), halfH: 4, wallH: 4.5, color: 0x999999, name: `dam${id++}` });

  // Breakwaters
  for (let i = 0; i < 5; i++) {
    const angle = (i / 5) * Math.PI * 2;
    const radius = W * 0.22;
    const ci = Math.round(W * 0.50 + Math.cos(angle) * radius);
    const cj = Math.round(H * 0.42 + Math.sin(angle) * radius);
    const hw = Math.round(4 + rng() * 3);
    if (ci - hw < 3 || ci + hw >= W - 3 || cj - 1 < 3 || cj + 1 >= H - 3) continue;
    obstacles.push({ cx: ci, cz: cj, halfW: hw, halfH: 1,
      wallH: 1.2 + rng() * 0.8, color: 0x777777, name: `breakwater${id++}` });
  }

  // Bollards
  for (let i = 0; i < 20; i++) {
    const ci = Math.round(15 + rng() * (W - 30));
    const cj = Math.round(15 + rng() * (H - 30));
    if (!canPlace(ci, cj, 1, 1)) continue;
    obstacles.push({ cx: ci, cz: cj, halfW: 1, halfH: 1,
      wallH: 0.4 + rng() * 0.3, color: 0x444444, name: `bollard${id++}` });
  }

  return obstacles;
}

/**
 * Composite terrain + all obstacles into a single bed heightmap.
 * Each cell = max(terrain, obstacle1_top, obstacle2_top, …).
 * Returns the composited bed array (W*H), ready for writeBedFull.
 */
function compositeBed(
  terrain: Float32Array, obstacles: Obstacle[], W: number, H: number,
): Float32Array {
  const bed = new Float32Array(terrain); // start with terrain as base
  for (const ob of obstacles) {
    const i0 = Math.max(0, ob.cx - ob.halfW);
    const i1 = Math.min(W - 1, ob.cx + ob.halfW);
    const j0 = Math.max(0, ob.cz - ob.halfH);
    const j1 = Math.min(H - 1, ob.cz + ob.halfH);
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const idx = j * W + i;
        const obstacleTop = terrain[idx] + ob.wallH;
        if (obstacleTop > bed[idx]) bed[idx] = obstacleTop;
      }
    }
  }
  return bed;
}

/**
 * Create visual meshes for all obstacles. Each mesh is a box positioned
 * so its base sits on the terrain and its top matches the composited bed.
 * Uses the SAME terrain + wallH values that compositeBed used.
 */
function spawnVisuals(
  ctx: DemoContext, obstacles: Obstacle[], terrain: Float32Array,
): void {
  const g = ctx.solver.grid;
  const W = g.width;
  const H = g.height;
  for (const ob of obstacles) {
    const worldX = g.origin[0] + ob.cx * g.dx;
    const worldZ = g.origin[1] + ob.cz * g.dx;
    const baseTerrain = (ob.cx >= 0 && ob.cx < W && ob.cz >= 0 && ob.cz < H)
      ? terrain[ob.cz * W + ob.cx] : 0;
    const spanW = 2 * ob.halfW + 1;
    const spanH = 2 * ob.halfH + 1;
    const sizeX = spanW * g.dx;
    const sizeZ = spanH * g.dx;
    const mat = new THREE.MeshStandardMaterial({ color: ob.color, roughness: 0.85 });
    const mesh = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(sizeX, ob.wallH, sizeZ), mat));
    mesh.name = ob.name;
    // Box base = terrain, box top = terrain + wallH (matches solver bed)
    mesh.position.set(worldX, baseTerrain + ob.wallH / 2, worldZ);
    ctx.scene.add(mesh);
  }
}

// ── Dam break state ──────────────────────────────────────────────────

interface DamBreakEvent {
  triggerTime: number;
  ci: number; cj: number;
  halfW: number; halfH: number;
  fired: boolean;
}

// ── Demo ─────────────────────────────────────────────────────────────

const demo: Demo = {
  id: '12-stress-test',
  label: 'L. Stress Test',
  description:
    'Kilometer-scale terrain stress test: complex mountain terrain with a large central lake, ' +
    '100+ buildings and obstacles, multiple flood sources pouring from mountain passes, ' +
    'periodic dam breaks creating massive waves, and floating debris. ' +
    'Pushes the SWE solver, rasterizer, and renderer to their limits.',

  setup(ctx) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;
    const dx = g.dx;
    const rng = makeRng(99999);

    // ── 1. Build terrain heightmap ──────────────────────────────────
    const terrain = new Float32Array(W * H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const wx = g.origin[0] + i * dx;
        const wz = g.origin[1] + j * dx;
        terrain[j * W + i] = sampleTerrain(wx, wz);
      }
    }

    // ── 2. Generate obstacles (pure data, no side effects) ───────
    const obstacles = generateObstacles(W, H, dx, g.origin[0], g.origin[1]);

    // ── 3. Composite bed = max(terrain, all obstacles) per cell ──
    // Single pass: every cell gets the highest of terrain and any
    // obstacle tops that cover it. Written to solver ONCE.
    const bed = compositeBed(terrain, obstacles, W, H);
    ctx.solver.writeBedFull(bed);

    // ── 4. Spawn visual meshes from same obstacle data ───────────
    // Positioned using terrain + wallH — identical to what compositeBed wrote.
    spawnVisuals(ctx, obstacles, terrain);

    // ── 5. Initialize water (fill the lake basin) ────────────────
    const water0 = new Float32Array(W * H);
    const LAKE_LEVEL = 1.2;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const bedH = bed[j * W + i];
        if (bedH < LAKE_LEVEL) {
          const wx = g.origin[0] + i * dx;
          const wz = g.origin[1] + j * dx;
          const distFromCenter = Math.sqrt(wx * wx + (wz + 0.5) * (wz + 0.5));
          if (distFromCenter < 4.0) {
            water0[j * W + i] = Math.min(LAKE_LEVEL - bedH, 2.0);
          }
        }
      }
    }
    ctx.solver.writeWaterFull(water0);

    // ── 4. Boundary conditions ─────────────────────────────────────
    // Multiple inflow sources from mountain passes
    // North pass (center) - major river source
    ctx.solver.writeBoundaryRegionTarget(
      { x: Math.round(W * 0.42), y: 0, w: Math.round(W * 0.16), h: 3 },
      BoundaryType.Inflow, 0,
    );
    // Northwest pass - glacier melt
    ctx.solver.writeBoundaryRegionTarget(
      { x: 0, y: 0, w: 3, h: Math.round(H * 0.20) },
      BoundaryType.Inflow, 0,
    );
    // Northeast pass - mountain spring
    ctx.solver.writeBoundaryRegionTarget(
      { x: W - 3, y: 0, w: 3, h: Math.round(H * 0.15) },
      BoundaryType.Inflow, 0,
    );

    // South outflow (drainage)
    ctx.solver.writeBoundaryRegionTarget(
      { x: Math.round(W * 0.30), y: H - 3, w: Math.round(W * 0.40), h: 3 },
      BoundaryType.Sea, 0,
    );
    // East outflow
    ctx.solver.writeBoundaryRegionTarget(
      { x: W - 3, y: Math.round(H * 0.60), w: 3, h: Math.round(H * 0.30) },
      BoundaryType.Sea, 0,
    );

    // ── 5. Terrain mesh for visual grounding ───────────────────────
    const terrainGeo = new THREE.PlaneGeometry(
      W * dx, H * dx,
      Math.min(W - 1, 255), Math.min(H - 1, 255),
    );
    terrainGeo.rotateX(-Math.PI / 2);
    const posAttr = terrainGeo.getAttribute('position') as THREE.BufferAttribute;
    const segW = Math.min(W - 1, 255);
    const segH = Math.min(H - 1, 255);
    for (let jj = 0; jj <= segH; jj++) {
      for (let ii = 0; ii <= segW; ii++) {
        const vi = jj * (segW + 1) + ii;
        const wx = posAttr.getX(vi);
        const wz = posAttr.getZ(vi);
        const h = sampleTerrain(wx, wz);
        posAttr.setY(vi, h);
      }
    }
    terrainGeo.computeVertexNormals();
    const terrainMat = new THREE.MeshStandardMaterial({
      color: 0x5a6e3a,
      roughness: 0.95,
      flatShading: true,
    });
    const terrainMesh = ownByDemo(new THREE.Mesh(terrainGeo, terrainMat));
    terrainMesh.name = 'terrain';
    ctx.scene.add(terrainMesh);

    // ── 6. Water surface ───────────────────────────────────────────
    const waterSurf = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(waterSurf.mesh));

    // ── 7. Camera: high and back to see the whole terrain ──────────
    ctx.camera.position.set(0, 18, 20);
    ctx.controls.target.set(0, 1, 0);
    ctx.controls.update();

    // ── 8. Dam break events ────────────────────────────────────────
    const damBreaks: DamBreakEvent[] = [
      // The main dam breaks at t=45s — water fills up, overtops, THEN it breaks
      {
        triggerTime: 45,
        ci: Math.round(W * 0.50), cj: Math.round(H * 0.68),
        halfW: Math.round(W * 0.35), halfH: 4,
        fired: false,
      },
      // A breakwater fails at t=60s
      {
        triggerTime: 60,
        ci: Math.round(W * 0.50 + Math.cos(0) * W * 0.22),
        cj: Math.round(H * 0.42 + Math.sin(0) * W * 0.22),
        halfW: Math.round(4 + 3 * 0.5), halfH: 1,
        fired: false,
      },
    ];

    // ── Store scratch state ────────────────────────────────────────
    ctx.scratch.water = waterSurf;
    ctx.scratch.t = 0;
    ctx.scratch.terrain = terrain;
    ctx.scratch.bed = bed;
    ctx.scratch.obstacles = obstacles;
    ctx.scratch.rng = rng;
    ctx.scratch.damBreaks = damBreaks;
    ctx.scratch.nextDebrisSpawn = 5;
    ctx.scratch.debrisCount = 0;
    ctx.scratch.obstacleCount = obstacles.length;
    ctx.scratch.nextImpact = 3;
  },

  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();

    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;
    const t = (ctx.scratch.t as number) + dt;
    ctx.scratch.t = t;
    const rng = ctx.scratch.rng as () => number;
    const terrain = ctx.scratch.terrain as Float32Array;

    // ── Inflow sources: ramp up over time ──────────────────────────
    // North pass: main river, ramps to 2.0m over 10s
    const northH = Math.min(2.0, 0.2 * t);
    ctx.solver.writeBoundaryRegionTarget(
      { x: Math.round(W * 0.42), y: 0, w: Math.round(W * 0.16), h: 3 },
      BoundaryType.Inflow, northH,
    );

    // NW glacier: slower, steady at 1.5m
    const nwH = Math.min(1.5, 0.15 * t);
    ctx.solver.writeBoundaryRegionTarget(
      { x: 0, y: 0, w: 3, h: Math.round(H * 0.20) },
      BoundaryType.Inflow, nwH,
    );

    // NE spring: pulsing (simulates rainfall bursts)
    const neBase = Math.min(1.0, 0.1 * t);
    const nePulse = neBase + 0.5 * Math.sin(t * 0.8) * Math.max(0, Math.min(1, (t - 5) / 5));
    ctx.solver.writeBoundaryRegionTarget(
      { x: W - 3, y: 0, w: 3, h: Math.round(H * 0.15) },
      BoundaryType.Inflow, Math.max(0, nePulse),
    );

    // ── Dam break events ───────────────────────────────────────────
    const damBreaks = ctx.scratch.damBreaks as DamBreakEvent[];
    const obstacles = ctx.scratch.obstacles as Obstacle[];
    for (const db of damBreaks) {
      if (!db.fired && t >= db.triggerTime) {
        db.fired = true;

        // Remove the obstacle from the list and find its scene name
        const removedIdx = obstacles.findIndex(
          (o) => o.cx === db.ci && o.cz === db.cj && o.halfW === db.halfW,
        );
        let removedName = '';
        if (removedIdx >= 0) {
          removedName = obstacles[removedIdx]!.name;
          obstacles.splice(removedIdx, 1);
        }

        // Recomposite the bed region WITHOUT the removed obstacle.
        // This correctly preserves any other overlapping obstacles.
        const spanW = 2 * db.halfW + 1;
        const spanH = 2 * db.halfH + 1;
        const i0 = Math.max(0, db.ci - db.halfW);
        const j0 = Math.max(0, db.cj - db.halfH);
        const i1 = Math.min(W - 1, db.ci + db.halfW);
        const j1 = Math.min(H - 1, db.cj + db.halfH);
        const rw = i1 - i0 + 1;
        const rh = j1 - j0 + 1;
        // Recomposite just this region from terrain + remaining obstacles
        const patch = new Float32Array(rw * rh);
        for (let jj = 0; jj < rh; jj++) {
          for (let ii = 0; ii < rw; ii++) {
            patch[jj * rw + ii] = terrain[(j0 + jj) * W + (i0 + ii)];
          }
        }
        for (const ob of obstacles) {
          const oi0 = Math.max(i0, ob.cx - ob.halfW);
          const oi1 = Math.min(i1, ob.cx + ob.halfW);
          const oj0 = Math.max(j0, ob.cz - ob.halfH);
          const oj1 = Math.min(j1, ob.cz + ob.halfH);
          for (let j = oj0; j <= oj1; j++) {
            for (let i = oi0; i <= oi1; i++) {
              const pi = (j - j0) * rw + (i - i0);
              const top = terrain[j * W + i] + ob.wallH;
              if (top > patch[pi]) patch[pi] = top;
            }
          }
        }
        ctx.solver.writeBedRegion({ x: i0, y: j0, w: rw, h: rh }, patch);

        // Remove visual mesh
        if (removedName) {
          const toRemove: THREE.Object3D[] = [];
          ctx.scene.traverse((obj) => { if (obj.name === removedName) toRemove.push(obj); });
          for (const obj of toRemove) {
            obj.removeFromParent();
            if (obj instanceof THREE.Mesh) {
              obj.geometry.dispose();
              (obj.material as THREE.Material).dispose();
            }
          }
        }
      }
    }

    // ── Periodic impact splashes (simulates rockfalls / large debris) ──
    const nextImpact = ctx.scratch.nextImpact as number;
    if (t >= nextImpact) {
      ctx.scratch.nextImpact = t + 1.5 + rng() * 3.0;

      // Random position in the lake area
      const impactWx = (rng() - 0.5) * 6;
      const impactWz = (rng() - 0.5) * 6 - 0.5;
      const impactBed = sampleTerrain(impactWx, impactWz);

      // Only splash if there's likely water
      if (impactBed < 1.5) {
        const result = applyImpact(ctx.solver, {
          worldX: impactWx,
          worldZ: impactWz,
          velocity: 6 + rng() * 8,
          shape: 'sphere' as const,
          size: 0.15 + rng() * 0.25,
          density: 2500,
          waterDepth: Math.max(0.1, 1.2 - impactBed),
        });
        spawnImpactSplash(ctx.splashes, {
          worldX: impactWx,
          worldZ: impactWz,
          velocity: 8 + rng() * 6,
          shape: 'sphere' as const,
          size: 0.2 + rng() * 0.2,
          density: 2500,
          waterDepth: Math.max(0.1, 1.2 - impactBed),
        }, result);
      }
    }

    // ── Spawn floating debris ──────────────────────────────────────
    const nextDebris = ctx.scratch.nextDebrisSpawn as number;
    const debrisCount = ctx.scratch.debrisCount as number;
    if (t >= nextDebris && debrisCount < 30) {
      ctx.scratch.nextDebrisSpawn = t + 2.0 + rng() * 4.0;
      ctx.scratch.debrisCount = debrisCount + 1;

      // Spawn near one of the inflow sources
      const source = rng() < 0.5 ? 0 : rng() < 0.5 ? 1 : 2;
      let spawnX: number, spawnZ: number;
      if (source === 0) {
        // North pass
        spawnX = g.origin[0] + W * 0.42 * g.dx + rng() * W * 0.16 * g.dx;
        spawnZ = g.origin[1] + 5 * g.dx;
      } else if (source === 1) {
        // NW
        spawnX = g.origin[0] + 5 * g.dx;
        spawnZ = g.origin[1] + rng() * H * 0.15 * g.dx;
      } else {
        // NE
        spawnX = g.origin[0] + (W - 5) * g.dx;
        spawnZ = g.origin[1] + rng() * H * 0.10 * g.dx;
      }

      const halfSide = 0.08 + rng() * 0.12;
      const spawnY = sampleTerrain(spawnX, spawnZ) + 1.5;
      const body = ctx.world.createRigidBody(
        ctx.rapier.RigidBodyDesc.dynamic()
          .setTranslation(spawnX, spawnY, spawnZ)
          .setLinearDamping(3.0)
          .setAngularDamping(3.0),
      );
      ctx.world.createCollider(
        ctx.rapier.ColliderDesc.cuboid(halfSide, halfSide * 0.5, halfSide)
          .setDensity(300 + rng() * 400),
        body,
      );
      const colors = [0xaa6633, 0x886644, 0x557733, 0x996644, 0x664422];
      const mesh = ownByDemo(
        new THREE.Mesh(
          new THREE.BoxGeometry(halfSide * 2, halfSide, halfSide * 2),
          new THREE.MeshStandardMaterial({ color: colors[Math.floor(rng() * colors.length)]! }),
        ),
      );
      ctx.scene.add(mesh);
      ctx.spawnCoupledBody({
        name: `debris_${debrisCount}`,
        body,
        halfExtents: [halfSide, halfSide * 0.5, halfSide],
        mesh,
        waterLevelRef: 1.2,
        bedLevelRef: 0,
      });
    }
  },
};

export default demo;
