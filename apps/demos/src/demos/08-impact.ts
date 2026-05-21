import * as THREE from 'three';
import type { Demo, DemoContext } from '../shared/Scene.js';
import { ownByDemo } from '../shared/Scene.js';
import { WaterSurface } from '../shared/Water.js';
import { applyImpact, spawnImpactSplash, type ImpactShape } from 'isenflow';
import RAPIER from '@dimforge/rapier3d-compat';

/**
 * Meteorite Impact — an endless rain of random objects strikes a deep lake.
 * Small debris makes ripples, large meteorites make tsunamis. Objects vary
 * in shape, size, density, drop height, and position. Uses the reusable
 * `applyImpact()` API.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ImpactorDef {
  name: string;
  shape: ImpactShape;
  size: number | readonly [number, number, number];
  density: number;
  dropHeight: number;
  dropTime: number;
  worldX: number;
  worldZ: number;
  color: number;
  emissive?: number;
}

interface ImpactorState extends ImpactorDef {
  spawned: boolean;
  impacted: boolean;
  fadeStart: number;
  mesh?: THREE.Mesh;
  body?: RAPIER.RigidBody;
}

// ---------------------------------------------------------------------------
// Random impactor generation
// ---------------------------------------------------------------------------

const WATER_DEPTH = 1.5;
const WORLD_HALF = 6.5; // stay inside 16m world minus walls

/** Weighted random pick: ~60% small, ~25% medium, ~10% large, ~5% meteorite. */
function randomCategory(rng: () => number): 'pebble' | 'medium' | 'large' | 'meteorite' {
  const r = rng();
  if (r < 0.60) return 'pebble';
  if (r < 0.85) return 'medium';
  if (r < 0.95) return 'large';
  return 'meteorite';
}

const COLORS = {
  pebble:    [0x887766, 0x998877, 0x776655, 0xaa9988],
  medium:    [0x555555, 0x666666, 0xaa7733, 0x888899],
  large:     [0x444444, 0x555566, 0x663322],
  meteorite: [0x331111, 0x441100, 0x220808],
};

let impactorCounter = 0;

function generateRandomImpactor(dropTime: number, rng: () => number): ImpactorDef {
  const cat = randomCategory(rng);
  const id = impactorCounter++;

  // Random position within the pool
  const worldX = (rng() - 0.5) * 2 * WORLD_HALF;
  const worldZ = (rng() - 0.5) * 2 * WORLD_HALF;

  // Pick shape: pebbles/meteorites always sphere, medium/large sometimes cuboid
  const useCuboid = (cat === 'medium' || cat === 'large') && rng() < 0.35;
  const shape: ImpactShape = useCuboid ? 'cuboid' : 'sphere';

  let size: number | readonly [number, number, number];
  let density: number;
  let dropHeight: number;
  let color: number;
  let emissive: number | undefined;

  const pick = <T>(arr: T[]) => arr[Math.floor(rng() * arr.length)]!;

  switch (cat) {
    case 'pebble': {
      const r = 0.04 + rng() * 0.08; // 0.04–0.12m
      size = r;
      density = 2000 + rng() * 1000;  // 2000–3000
      dropHeight = 4 + rng() * 4;     // 4–8m
      color = pick(COLORS.pebble);
      break;
    }
    case 'medium': {
      if (shape === 'cuboid') {
        const hx = 0.1 + rng() * 0.2;
        const hy = 0.1 + rng() * 0.3;
        const hz = 0.1 + rng() * 0.2;
        size = [hx, hy, hz] as const;
      } else {
        size = 0.15 + rng() * 0.25; // 0.15–0.40m
      }
      density = 1500 + rng() * 3000;  // 1500–4500 (wood to stone)
      dropHeight = 6 + rng() * 8;     // 6–14m
      color = pick(COLORS.medium);
      break;
    }
    case 'large': {
      if (shape === 'cuboid') {
        const hx = 0.2 + rng() * 0.3;
        const hy = 0.2 + rng() * 0.5;
        const hz = 0.2 + rng() * 0.3;
        size = [hx, hy, hz] as const;
      } else {
        size = 0.4 + rng() * 0.3; // 0.4–0.7m
      }
      density = 2500 + rng() * 2000;  // 2500–4500
      dropHeight = 10 + rng() * 8;    // 10–18m
      color = pick(COLORS.large);
      break;
    }
    case 'meteorite': {
      size = 0.7 + rng() * 0.5; // 0.7–1.2m
      density = 5000 + rng() * 4000;  // 5000–9000 (iron-nickel)
      dropHeight = 15 + rng() * 10;   // 15–25m
      color = pick(COLORS.meteorite);
      emissive = 0x330000;
      break;
    }
  }

  return {
    name: `${cat}_${id}`,
    shape,
    size,
    density,
    dropHeight,
    dropTime,
    worldX,
    worldZ,
    color,
    emissive,
  };
}

// ---------------------------------------------------------------------------
// Mesh / collider helpers
// ---------------------------------------------------------------------------

function effectiveRadius(def: ImpactorDef): number {
  return typeof def.size === 'number' ? def.size : Math.max(...(def.size as readonly [number, number, number]));
}

function createMesh(def: ImpactorDef): THREE.Mesh {
  const mat = new THREE.MeshStandardMaterial({
    color: def.color,
    roughness: 0.9,
    ...(def.emissive ? { emissive: def.emissive, emissiveIntensity: 0.5 } : {}),
  });
  if (def.shape === 'sphere') {
    const r = def.size as number;
    return new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), mat);
  }
  const [hx, hy, hz] = def.size as readonly [number, number, number];
  return new THREE.Mesh(new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2), mat);
}

function createCollider(ctx: DemoContext, def: ImpactorDef): RAPIER.ColliderDesc {
  if (def.shape === 'sphere') {
    return ctx.rapier.ColliderDesc.ball(def.size as number).setDensity(def.density);
  }
  const [hx, hy, hz] = def.size as readonly [number, number, number];
  return ctx.rapier.ColliderDesc.cuboid(hx, hy, hz).setDensity(def.density);
}

// ---------------------------------------------------------------------------
// Seeded PRNG (xorshift32) for reproducible randomness
// ---------------------------------------------------------------------------

function makeRng(seed: number): () => number {
  let s = seed | 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

/** Max simultaneous in-flight + fading impactors (to limit Rapier body count). */
const MAX_ACTIVE = 12;

const demo: Demo = {
  id: '08-impact',
  label: 'H. Meteorite Impact',
  description:
    'An endless rain of random objects — pebbles, crates, rocks, and meteorites — strikes a deep lake. Wave intensity scales with mass, velocity, and shape.',

  setup(ctx) {
    const g = ctx.solver.grid;
    const W = g.width;
    const H = g.height;
    const dx = g.dx;

    // Deep pool
    ctx.solver.writeWaterFull(new Float32Array(W * H).fill(WATER_DEPTH));

    // Perimeter walls
    const WALL_H = 3.0;
    const T = 2;
    const fillBed = (x: number, y: number, w: number, h: number, elev: number) => {
      ctx.solver.writeBedRegion({ x, y, w, h }, new Float32Array(w * h).fill(elev));
      ctx.solver.writeBoundaryRegion({ x, y, w, h }, 1);
    };
    fillBed(0, 0, W, T, WALL_H);
    fillBed(0, H - T, W, T, WALL_H);
    fillBed(0, 0, T, H, WALL_H);
    fillBed(W - T, 0, T, H, WALL_H);

    // Visual walls
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x607080, roughness: 0.8 });
    const worldW = W * dx;
    const worldH = H * dx;
    const bt = T * dx;
    const mkWall = (x: number, z: number, lx: number, lz: number) => {
      const m = ownByDemo(new THREE.Mesh(new THREE.BoxGeometry(lx, WALL_H, lz), wallMat));
      m.position.set(x, WALL_H / 2, z);
      ctx.scene.add(m);
    };
    mkWall(0, g.origin[1] + bt / 2, worldW, bt);
    mkWall(0, g.origin[1] + worldH - bt / 2, worldW, bt);
    mkWall(g.origin[0] + bt / 2, 0, bt, worldH);
    mkWall(g.origin[0] + worldW - bt / 2, 0, bt, worldH);

    // Floor collider
    const floorBody = ctx.world.createRigidBody(
      ctx.rapier.RigidBodyDesc.fixed().setTranslation(0, -0.05, 0),
    );
    ctx.world.createCollider(
      ctx.rapier.ColliderDesc.cuboid(worldW / 2, 0.05, worldH / 2),
      floorBody,
    );

    // Water surface
    const water = new WaterSurface(ctx.solver);
    ctx.scene.add(ownByDemo(water.mesh));

    impactorCounter = 0;
    const rng = makeRng(42);

    // Pre-schedule the initial scripted sequence, then switch to random
    const impactors: ImpactorState[] = [];

    // First 7 are the curated showcase (same as before)
    const scripted: ImpactorDef[] = [
      { name: 'pebbleA',   shape: 'sphere', size: 0.08,            density: 2500, dropHeight: 6,  dropTime: 0.5, worldX: -3, worldZ: -2, color: 0x887766 },
      { name: 'pebbleB',   shape: 'sphere', size: 0.06,            density: 2500, dropHeight: 5,  dropTime: 1.5, worldX: 2,  worldZ: 3,  color: 0x998877 },
      { name: 'crate',     shape: 'cuboid', size: [0.2, 0.2, 0.2], density: 800,  dropHeight: 8,  dropTime: 3.0, worldX: 3,  worldZ: -1, color: 0xaa7733 },
      { name: 'rock',      shape: 'sphere', size: 0.30,            density: 2800, dropHeight: 10, dropTime: 4.5, worldX: -1, worldZ: 1,  color: 0x555555 },
      { name: 'steelBeam', shape: 'cuboid', size: [0.1, 0.5, 0.1], density: 7800, dropHeight: 10, dropTime: 6.0, worldX: -3, worldZ: 2,  color: 0x888899 },
      { name: 'boulder',   shape: 'sphere', size: 0.50,            density: 3000, dropHeight: 12, dropTime: 7.5, worldX: 2,  worldZ: -2, color: 0x444444 },
      { name: 'meteorite', shape: 'sphere', size: 0.90,            density: 7800, dropHeight: 18, dropTime: 10.0, worldX: 0, worldZ: 0,  color: 0x331111, emissive: 0x330000 },
    ];
    for (const d of scripted) {
      impactors.push({ ...d, spawned: false, impacted: false, fadeStart: 0 });
    }

    ctx.scratch.water = water;
    ctx.scratch.t = 0;
    ctx.scratch.impactors = impactors;
    ctx.scratch.rng = rng;
    ctx.scratch.nextDropTime = 12; // first random drop after scripted sequence
  },

  tick(ctx, dt) {
    (ctx.scratch.water as WaterSurface).update();
    ctx.scratch.t = (ctx.scratch.t as number) + dt;
    const t = ctx.scratch.t as number;
    const impactors = ctx.scratch.impactors as ImpactorState[];
    const rng = ctx.scratch.rng as () => number;

    // Schedule new random drops
    if (t >= (ctx.scratch.nextDropTime as number)) {
      const activeCount = impactors.filter((i) => i.spawned && (!i.impacted || i.mesh)).length;
      if (activeCount < MAX_ACTIVE) {
        const def = generateRandomImpactor(t, rng);
        impactors.push({ ...def, spawned: false, impacted: false, fadeStart: 0 });
        // Random interval: 0.8–2.5s for small stuff, longer gaps after meteorites
        const interval = 0.8 + rng() * 1.7;
        ctx.scratch.nextDropTime = t + interval;
      } else {
        // Too many active, retry soon
        ctx.scratch.nextDropTime = t + 0.3;
      }
    }

    for (const imp of impactors) {
      // Phase A: Spawn at drop time
      if (!imp.spawned && t >= imp.dropTime) {
        imp.spawned = true;

        const mesh = ownByDemo(createMesh(imp));
        mesh.name = imp.name;
        mesh.position.set(imp.worldX, imp.dropHeight, imp.worldZ);
        ctx.scene.add(mesh);
        imp.mesh = mesh;

        const body = ctx.world.createRigidBody(
          ctx.rapier.RigidBodyDesc.dynamic()
            .setTranslation(imp.worldX, imp.dropHeight, imp.worldZ)
            .setLinearDamping(0.1),
        );
        ctx.world.createCollider(createCollider(ctx, imp), body);
        imp.body = body;
      }

      // Phase B: Falling — sync mesh, detect water surface hit
      if (imp.spawned && !imp.impacted && imp.body) {
        const pos = imp.body.translation();
        if (imp.mesh) imp.mesh.position.set(pos.x, pos.y, pos.z);

        const r = effectiveRadius(imp);
        if (pos.y <= WATER_DEPTH + r * 0.5) {
          imp.impacted = true;
          imp.fadeStart = t;

          const vel = imp.body.linvel();
          const impactVelocity = Math.abs(vel.y);

          const result = applyImpact(ctx.solver, {
            worldX: pos.x,
            worldZ: pos.z,
            velocity: impactVelocity,
            shape: imp.shape,
            size: imp.size,
            density: imp.density,
            waterDepth: WATER_DEPTH,
          });

          spawnImpactSplash(ctx.splashes, {
            worldX: pos.x,
            worldZ: pos.z,
            velocity: impactVelocity,
            shape: imp.shape,
            size: imp.size,
            density: imp.density,
            waterDepth: WATER_DEPTH,
          }, result);

          ctx.world.removeRigidBody(imp.body);
          imp.body = undefined;

          if (imp.mesh) {
            const m = imp.mesh.material as THREE.MeshStandardMaterial;
            m.transparent = true;
          }
        }
      }

      // Phase C: Post-impact mesh fade
      if (imp.impacted && imp.mesh) {
        const fadeElapsed = t - imp.fadeStart;
        const opacity = Math.max(0, 1 - fadeElapsed / 0.5);
        const m = imp.mesh.material as THREE.MeshStandardMaterial;
        m.opacity = opacity;
        imp.mesh.position.y -= dt * 2;
        if (opacity <= 0) {
          ctx.scene.remove(imp.mesh);
          imp.mesh = undefined;
        }
      }
    }

    // Prune fully-done impactors to avoid unbounded list growth
    const active = ctx.scratch.impactors as ImpactorState[];
    if (active.length > 50) {
      ctx.scratch.impactors = active.filter((i) => !i.impacted || i.mesh != null);
    }
  },
};

export default demo;
