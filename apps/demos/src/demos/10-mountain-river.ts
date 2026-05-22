import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { BoundaryType, applyImpact, spawnImpactSplash } from 'isenflow';

/**
 * Mountain River — a sprawling alpine landscape with multiple water features,
 * periodic weather events, click-to-interact rain/rockfall, and endless
 * flowing water dynamics you can watch for hours.
 *
 * Features:
 * - Rich multi-peak terrain with valleys, ridges, a lake basin, and gorge
 * - 3 independent water sources: glacier melt (NW), mountain spring (NE), rain
 * - A mid-altitude lake that fills, overflows, and cascades into the gorge
 * - Periodic "weather cycles": calm → rain → storm → calm (60s loops)
 * - Click anywhere to trigger a rockfall splash or rain burst
 * - Floating debris spawned periodically, carried by currents
 * - Open outflow on south edge for endless drainage
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

// Complex alpine landscape
const FEATURES: Peak[] = [
  // ── Main mountain range (NE to center) ──
  { cx:  3.5, cz: -4.0, amp: 4.2, sx: 1.6, sz: 1.8 },  // North peak (tallest)
  { cx:  4.5, cz: -1.5, amp: 3.6, sx: 1.4, sz: 1.6 },  // Middle peak
  { cx:  3.0, cz:  1.5, amp: 3.0, sx: 1.5, sz: 1.3 },  // South peak
  // Ridge connecting the range
  { cx:  3.8, cz: -2.8, amp: 2.2, sx: 0.8, sz: 1.0 },
  { cx:  3.8, cz:  0.0, amp: 1.8, sx: 0.7, sz: 1.2 },

  // ── Western highlands ──
  { cx: -4.5, cz: -3.5, amp: 3.2, sx: 1.8, sz: 1.5 },  // NW glacier peak
  { cx: -3.0, cz: -1.0, amp: 2.0, sx: 1.5, sz: 1.2 },  // W foothill

  // ── Lake basin rim (mid-south, forms a natural dam) ──
  { cx: -1.0, cz:  2.5, amp: 1.6, sx: 2.5, sz: 0.6 },  // Dam ridge (E-W)
  { cx: -3.5, cz:  1.5, amp: 2.4, sx: 1.0, sz: 1.5 },  // W rim
  { cx:  1.5, cz:  2.0, amp: 1.8, sx: 0.8, sz: 1.2 },  // E rim

  // ── Southern foothills ──
  { cx: -2.0, cz:  5.5, amp: 1.2, sx: 1.5, sz: 1.0 },
  { cx:  2.5, cz:  5.0, amp: 1.0, sx: 1.2, sz: 1.0 },
  { cx:  0.0, cz:  6.5, amp: 0.6, sx: 1.5, sz: 0.8 },

  // ── Small terrain bumps for texture ──
  { cx: -6.0, cz:  0.0, amp: 0.7, sx: 0.8, sz: 0.8 },
  { cx:  6.0, cz: -5.5, amp: 0.5, sx: 0.7, sz: 0.7 },
  { cx: -5.0, cz:  4.0, amp: 0.5, sx: 0.6, sz: 0.6 },
];

// River/valley channels carved into terrain
const CHANNELS: Peak[] = [
  // ── Main river: NW glacier → central valley → lake ──
  { cx: -3.0, cz: -5.5, amp: -0.7, sx: 0.5, sz: 1.2 },
  { cx: -2.0, cz: -3.5, amp: -0.8, sx: 0.5, sz: 1.0 },
  { cx: -1.0, cz: -1.5, amp: -0.9, sx: 0.5, sz: 1.0 },
  { cx: -0.5, cz:  0.5, amp: -1.0, sx: 0.6, sz: 1.0 },

  // ── NE stream: spring → joins main river ──
  { cx:  2.0, cz: -5.5, amp: -0.5, sx: 0.4, sz: 1.0 },
  { cx:  1.0, cz: -3.5, amp: -0.6, sx: 0.4, sz: 1.0 },
  { cx:  0.3, cz: -2.0, amp: -0.7, sx: 0.5, sz: 0.8 },

  // ── Lake basin (wide depression) ──
  { cx: -0.5, cz:  1.0, amp: -0.8, sx: 1.8, sz: 1.0 },

  // ── Gorge: lake overflow → south outflow ──
  { cx:  0.0, cz:  3.5, amp: -0.6, sx: 0.4, sz: 0.8 },
  { cx:  0.2, cz:  5.0, amp: -0.5, sx: 0.5, sz: 1.0 },
  { cx:  0.0, cz:  6.5, amp: -0.4, sx: 0.6, sz: 1.0 },
];

function sampleTerrain(wx: number, wz: number): number {
  let h = 0;
  for (const p of FEATURES) {
    const dx = wx - p.cx;
    const dz = wz - p.cz;
    h += p.amp * Math.exp(-(dx * dx) / (2 * p.sx * p.sx) - (dz * dz) / (2 * p.sz * p.sz));
  }
  for (const p of CHANNELS) {
    const dx = wx - p.cx;
    const dz = wz - p.cz;
    h += p.amp * Math.exp(-(dx * dx) / (2 * p.sx * p.sx) - (dz * dz) / (2 * p.sz * p.sz));
  }
  return Math.max(0, Math.min(h, 5));
}

// ── Color ramp ───────────────────────────────────────────────────────

const COLOR_STOPS: [number, THREE.Color][] = [
  [0.0, new THREE.Color(0x1a4a12)],  // deep green – riverbed
  [0.2, new THREE.Color(0x2d6a1e)],  // green – valley floor
  [0.6, new THREE.Color(0x4a8a2e)],  // bright green – meadow
  [1.2, new THREE.Color(0x7a7a3a)],  // olive – treeline
  [2.0, new THREE.Color(0x8b7355)],  // brown – alpine
  [3.0, new THREE.Color(0x8a8a80)],  // gray – rocky
  [3.8, new THREE.Color(0xb0b0a8)],  // light gray – high altitude
  [4.5, new THREE.Color(0xe8e8e8)],  // near-white – snow
];

function terrainColor(h: number): THREE.Color {
  for (let i = 1; i < COLOR_STOPS.length; i++) {
    if (h <= COLOR_STOPS[i][0]) {
      const [h0, c0] = COLOR_STOPS[i - 1];
      const [h1, c1] = COLOR_STOPS[i];
      const t = (h - h0) / (h1 - h0);
      return new THREE.Color().lerpColors(c0, c1, t);
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1][1].clone();
}

// ── Inflow source definitions ────────────────────────────────────────

interface WaterSource {
  /** Grid cell center */
  ci: number; cj: number;
  /** Radius in cells */
  radius: number;
  /** Base depth (m) */
  baseDepth: number;
  /** Multiplier from weather (applied in tick) */
  weatherScale: number;
}

// ── Weather system ───────────────────────────────────────────────────

const WEATHER_CYCLE = 90;  // seconds per full cycle

function weatherIntensity(t: number): number {
  // Smooth cycling: calm → building → storm → receding → calm
  const phase = (t % WEATHER_CYCLE) / WEATHER_CYCLE;
  // Two storm peaks per cycle for variety
  const storm1 = Math.exp(-((phase - 0.3) ** 2) / 0.005);
  const storm2 = Math.exp(-((phase - 0.7) ** 2) / 0.008) * 0.6;
  return Math.min(1, storm1 + storm2);
}

// ── Debris types ─────────────────────────────────────────────────────

interface DebrisTemplate {
  halfExtents: readonly [number, number, number];
  density: number;
  color: number;
  name: string;
}

const DEBRIS_TYPES: DebrisTemplate[] = [
  { halfExtents: [0.25, 0.04, 0.04], density: 550, color: 0x8b5a2b, name: 'log' },
  { halfExtents: [0.15, 0.08, 0.12], density: 600, color: 0x6b4226, name: 'branch' },
  { halfExtents: [0.08, 0.08, 0.08], density: 450, color: 0xa0763a, name: 'crate' },
  { halfExtents: [0.12, 0.05, 0.10], density: 500, color: 0x7a5230, name: 'plank' },
];

const MAX_DEBRIS = 16;

// ── Rain drop visual helper ──────────────────────────────────────────

interface RainDrop {
  ci: number; cj: number;
  radius: number;
  depth: number;
}

// ── Demo ─────────────────────────────────────────────────────────────

const HF_RES = 64;

const demo: Demo = {
  id: '10-mountain-river',
  label: 'J. Mountain River',
  description:
    'Alpine landscape with glaciers, springs, a mountain lake, and a gorge. ' +
    'Weather cycles bring rain and storms every ~90s. ' +
    'Click anywhere to drop a boulder into the water. ' +
    'Floating debris rides the currents endlessly.',

  setup(ctx: DemoContext) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;
    const dx = g.dx;
    const rng = makeRng(2024);

    // ── 1. Generate terrain heightmap ──
    const terrain = new Float32Array(W * H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const wx = g.origin[0] + i * dx;
        const wz = g.origin[1] + j * dx;
        terrain[j * W + i] = sampleTerrain(wx, wz);
      }
    }
    ctx.solver.writeBedFull(terrain);

    // ── 2. Rapier heightfield collider ──
    const nR = HF_RES + 1;
    const rapierHeights = new Float32Array(nR * nR);
    for (let col = 0; col < nR; col++) {
      for (let row = 0; row < nR; row++) {
        const si = Math.min(Math.round(col * (W - 1) / HF_RES), W - 1);
        const sj = Math.min(Math.round(row * (H - 1) / HF_RES), H - 1);
        rapierHeights[col * nR + row] = terrain[sj * W + si];
      }
    }
    const hfBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.fixed().setTranslation(0, 0, 0),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.heightfield(
        HF_RES, HF_RES, rapierHeights,
        { x: 16, y: 1, z: 16 } as any,
      ),
      hfBody,
    );

    // ── 3. Terrain mesh ──
    const terrainGeom = new THREE.PlaneGeometry(W * dx, H * dx, W, H);
    terrainGeom.rotateX(-Math.PI / 2);
    const pos = terrainGeom.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    for (let idx = 0; idx < pos.count; idx++) {
      const px = pos.getX(idx);
      const pz = pos.getZ(idx);
      const h = sampleTerrain(px, pz);
      pos.setY(idx, h);
      terrainColor(h).toArray(colors, idx * 3);
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

    // ── 4. Water sources (boundary-driven inflow) ──
    // Glacier melt: NW corner — steady, increases in storms
    const glacierSource: WaterSource = {
      ci: Math.round((-4.5 - g.origin[0]) / dx),
      cj: 0,
      radius: Math.round(1.5 / dx),
      baseDepth: 0.6,
      weatherScale: 1.0,
    };
    // Mountain spring: NE area — intermittent bursts
    const springSource: WaterSource = {
      ci: Math.round((2.0 - g.origin[0]) / dx),
      cj: 0,
      radius: Math.round(1.0 / dx),
      baseDepth: 0.3,
      weatherScale: 1.0,
    };

    // Set up north-edge inflow zones
    const applySource = (src: WaterSource, depth: number) => {
      const x0 = Math.max(0, src.ci - src.radius);
      const x1 = Math.min(W, src.ci + src.radius);
      for (let row = 0; row < 2; row++) {
        ctx.solver.writeBoundaryRegionTarget(
          { x: x0, y: row, w: x1 - x0, h: 1 },
          BoundaryType.Inflow,
          depth,
        );
      }
    };

    applySource(glacierSource, 0);
    applySource(springSource, 0);

    // ── 5. South outflow (Sea boundary for active drainage) ──
    for (let row = 0; row < 2; row++) {
      ctx.solver.writeBoundaryRegionTarget(
        { x: 0, y: H - 1 - row, w: W, h: 1 },
        BoundaryType.Sea,
        0,
      );
    }
    // East/West: closed
    for (let j = 2; j < H - 2; j++) {
      ctx.solver.writeBoundaryRegionTarget({ x: 0, y: j, w: 1, h: 1 }, BoundaryType.Closed, 0);
      ctx.solver.writeBoundaryRegionTarget({ x: W - 1, y: j, w: 1, h: 1 }, BoundaryType.Closed, 0);
    }

    // ── 6. Initial water: seed the lake basin + river ──
    const water0 = new Float32Array(W * H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const bed = terrain[j * W + i];
        const wx = g.origin[0] + i * dx;
        const wz = g.origin[1] + j * dx;
        // Lake basin: deeper initial fill
        const distToLake = Math.hypot(wx - (-0.5), wz - 1.0);
        if (distToLake < 2.5 && bed < 0.8) {
          water0[j * W + i] = Math.max(0, 0.8 - bed);
        } else if (bed < 0.3) {
          // River channels: thin layer
          water0[j * W + i] = 0.1;
        }
      }
    }
    ctx.solver.writeWaterFull(water0);

    // ── 7. Water surface renderer ──
    const waterSurf = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(waterSurf.mesh));

    // ── 8. Click-to-interact: raycaster for boulder drops ──
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
        // Apply water impact at click location (like a boulder splash)
        const result = applyImpact(ctx.solver, {
          worldX: pt.x,
          worldZ: pt.z,
          velocity: 8 + rng() * 6,
          shape: 'sphere',
          size: 0.3 + rng() * 0.4,
          density: 2500,
          waterDepth: 0.5,
        });
        spawnImpactSplash(ctx.splashes, {
          worldX: pt.x,
          worldZ: pt.z,
          velocity: 10,
          shape: 'sphere',
          size: 0.35,
          density: 2500,
          waterDepth: 0.5,
        }, result);
      }
    };
    ctx.renderer.domElement.addEventListener('pointerdown', onPointerDown);

    // ── 9. Camera ──
    ctx.camera.position.set(-8, 14, 12);
    ctx.controls.target.set(0, 0, 0);
    ctx.controls.update();

    // ── Scratch state ──
    ctx.scratch.water = waterSurf;
    ctx.scratch.t = 0;
    ctx.scratch.terrain = terrain;
    ctx.scratch.rng = rng;
    ctx.scratch.glacierSource = glacierSource;
    ctx.scratch.springSource = springSource;
    ctx.scratch.debrisCount = 0;
    ctx.scratch.nextDebrisTime = 5;
    ctx.scratch.nextRainTime = 3;
    ctx.scratch.pendingRestores = [] as Array<{ at: number; region: { x: number; y: number; w: number; h: number }; buf: Float32Array }>;
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

    // ── Weather intensity ──
    const weather = weatherIntensity(t);

    // ── Update water sources based on weather ──
    const glacierSource = ctx.scratch.glacierSource as WaterSource;
    const springSource = ctx.scratch.springSource as WaterSource;

    // Glacier: steady base + weather boost, slow ramp
    const glacierDepth = glacierSource.baseDepth + weather * 1.2;

    // Spring: pulsing (bursts every ~8s) + weather modulation
    const springPulse = 0.5 + 0.5 * Math.sin(t * 0.8);
    const springDepth = springSource.baseDepth * springPulse + weather * 0.8;

    const applySource = (src: WaterSource, depth: number) => {
      const x0 = Math.max(0, src.ci - src.radius);
      const x1 = Math.min(W, src.ci + src.radius);
      for (let row = 0; row < 2; row++) {
        ctx.solver.writeBoundaryRegionTarget(
          { x: x0, y: row, w: x1 - x0, h: 1 },
          BoundaryType.Inflow,
          depth,
        );
      }
    };
    applySource(glacierSource, glacierDepth);
    applySource(springSource, springDepth);

    // ── Rain: bed-piston pulses at random low-altitude points ──
    // Temporarily raise bed in small patches to push water up (like raindrops
    // displacing the surface). The solver propagates the displaced water as waves.
    if (weather > 0.2 && t >= (ctx.scratch.nextRainTime as number)) {
      const dropCount = Math.floor(2 + weather * 10);
      for (let d = 0; d < dropCount; d++) {
        const ri = Math.floor(rng() * (W - 8)) + 4;
        const rj = Math.floor(rng() * (H - 8)) + 4;
        const bedH = terrain[rj * W + ri];
        if (bedH < 1.2) {
          const R = 2;
          const diam = R * 2 + 1;
          const pulse = 0.04 + weather * 0.12; // bed rise in meters
          const buf = new Float32Array(diam * diam);
          for (let dj = 0; dj < diam; dj++) {
            for (let di = 0; di < diam; di++) {
              const dr = Math.hypot(di - R, dj - R) / R;
              // Cosine bell on top of existing terrain
              const base = terrain[(rj - R + dj) * W + (ri - R + di)] ?? 0;
              buf[dj * diam + di] = dr < 1.0
                ? base + pulse * 0.5 * (1 + Math.cos(Math.PI * dr))
                : base;
            }
          }
          ctx.solver.writeBedRegion(
            { x: ri - R, y: rj - R, w: diam, h: diam },
            buf,
          );
          // Schedule removal: restore original bed after a short delay
          const restoreBuf = new Float32Array(diam * diam);
          for (let dj = 0; dj < diam; dj++) {
            for (let di = 0; di < diam; di++) {
              restoreBuf[dj * diam + di] = terrain[(rj - R + dj) * W + (ri - R + di)] ?? 0;
            }
          }
          const restoreAt = t + 0.08;
          const restoreRegion = { x: ri - R, y: rj - R, w: diam, h: diam };
          const pending = ctx.scratch.pendingRestores as Array<{ at: number; region: typeof restoreRegion; buf: Float32Array }>;
          pending.push({ at: restoreAt, region: restoreRegion, buf: restoreBuf });
        }
      }
      ctx.scratch.nextRainTime = t + 0.15 + (1 - weather) * 0.5;
    }

    // ── Process pending bed restores (rain piston cleanup) ──
    const pending = ctx.scratch.pendingRestores as Array<{ at: number; region: { x: number; y: number; w: number; h: number }; buf: Float32Array }>;
    for (let i = pending.length - 1; i >= 0; i--) {
      if (t >= pending[i].at) {
        ctx.solver.writeBedRegion(pending[i].region, pending[i].buf);
        pending.splice(i, 1);
      }
    }

    // ── Periodic debris spawning ──
    if (t >= (ctx.scratch.nextDebrisTime as number)) {
      const debrisCount = ctx.scratch.debrisCount as number;
      if (debrisCount < MAX_DEBRIS) {
        const tmpl = DEBRIS_TYPES[Math.floor(rng() * DEBRIS_TYPES.length)];
        const [hx, hy, hz] = tmpl.halfExtents;

        // Spawn near one of the inflow sources
        const useGlacier = rng() > 0.4;
        const srcCi = useGlacier ? glacierSource.ci : springSource.ci;
        const spawnWx = g.origin[0] + srcCi * dx + (rng() - 0.5) * 1.0;
        const spawnWz = g.origin[1] + 3 * dx; // just inside north edge
        const bedH = sampleTerrain(spawnWx, spawnWz);

        const mesh = ownByDemo(
          new THREE.Mesh(
            new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2),
            new THREE.MeshStandardMaterial({ color: tmpl.color, roughness: 0.85 }),
          ),
        );
        mesh.name = `${tmpl.name}_${debrisCount}`;
        ctx.scene.add(mesh);

        const body = ctx.world.createRigidBody(
          ctx.rapier.RigidBodyDesc.dynamic()
            .setTranslation(spawnWx, bedH + 0.5, spawnWz)
            .setLinearDamping(1.5)
            .setAngularDamping(3.0),
        );
        ctx.world.createCollider(
          ctx.rapier.ColliderDesc.cuboid(hx, hy, hz).setDensity(tmpl.density),
          body,
        );
        ctx.spawnCoupledBody({
          name: mesh.name,
          body,
          halfExtents: [hx, hy, hz],
          mesh,
          waterLevelRef: 0.4,
          bedLevelRef: bedH,
        });

        ctx.scratch.debrisCount = debrisCount + 1;
      }
      // Stagger spawns: 4-12s apart
      ctx.scratch.nextDebrisTime = t + 4 + rng() * 8;
    }

    // ── Remove debris that exits the south boundary ──
    for (const [id, cb] of ctx.coupledBodies) {
      const pos = cb.body.translation();
      if (pos.z > 8.5 || pos.y < -2) {
        if (cb.mesh) ctx.scene.remove(cb.mesh);
        ctx.world.removeRigidBody(cb.body);
        ctx.coupledBodies.delete(id);
        ctx.scratch.debrisCount = Math.max(0, (ctx.scratch.debrisCount as number) - 1);
      }
    }
  },

  cleanup(ctx: DemoContext) {
    const handler = ctx.scratch.onPointerDown as EventListener;
    if (handler) {
      ctx.renderer.domElement.removeEventListener('pointerdown', handler);
    }
  },
};

export default demo;
