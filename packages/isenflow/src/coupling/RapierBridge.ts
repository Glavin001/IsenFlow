/**
 * Applies decoded ChunkForce values to Rapier rigid bodies.
 *
 * `@dimforge/rapier3d-compat` is a peer dependency: we type-only-import here
 * to avoid the dependency in pure-math test runs.
 */
import type { ChunkForce } from './ForceReadback.js';
import { computeCpuBuoyancy, computeVerticalDamping, estimateFootprintCells } from './CpuBuoyancy.js';

/** Minimal duck type of `RigidBody` we depend on. */
export interface RapierBodyLike {
  resetForces(wakeUp: boolean): void;
  resetTorques(wakeUp: boolean): void;
  addForce(force: { x: number; y: number; z: number }, wakeUp: boolean): void;
  addTorque(torque: { x: number; y: number; z: number }, wakeUp: boolean): void;
}

/** Extended duck type with velocity access, needed for stabilized coupling. */
export interface RapierBodyFull extends RapierBodyLike {
  linvel(): { x: number; y: number; z: number };
  setLinvel(v: { x: number; y: number; z: number }, wakeUp: boolean): void;
  translation(): { x: number; y: number; z: number };
}

/** Map keyed by chunk id (0 reserved). */
export type RapierBodyMap = ReadonlyMap<number, RapierBodyLike>;

/** Sanity bound on a single per-frame force contribution. */
const MAX_FORCE_NEWTONS = 5000;
const MAX_TORQUE_NM = 5000;

function clamp(v: number, lim: number): number {
  if (!Number.isFinite(v)) return 0;
  return v > lim ? lim : v < -lim ? -lim : v;
}

export interface ApplyForcesOptions {
  /** Multiplicative scaling on accumulated forces (e.g. for tuning). */
  scale?: number;
  /** Skip the resetForces/resetTorques calls; useful when stacking. */
  noReset?: boolean;
}

/**
 * Apply per-chunk forces & torques to bodies, indexed by chunk id.
 *
 * Accepts either:
 *   - a Map<chunkId, body>, OR
 *   - a sparse array indexed by chunk id (slot 0 ignored)
 */
export function applyForcesToBodies(
  bodies: RapierBodyMap | ReadonlyArray<RapierBodyLike | null | undefined>,
  forces: ReadonlyArray<ChunkForce>,
  opts: ApplyForcesOptions = {},
): void {
  const scale = opts.scale ?? 1;
  const visit = (chunkId: number, body: RapierBodyLike): void => {
    if (chunkId <= 0) return;
    const f = forces[chunkId];
    if (!f) return;
    if (!opts.noReset) {
      body.resetForces(true);
      body.resetTorques(true);
    }
    body.addForce(
      {
        x: clamp(f.fx * scale, MAX_FORCE_NEWTONS),
        y: clamp(f.fy * scale, MAX_FORCE_NEWTONS),
        z: clamp(f.fz * scale, MAX_FORCE_NEWTONS),
      },
      true,
    );
    body.addTorque(
      {
        x: clamp(f.tx * scale, MAX_TORQUE_NM),
        y: clamp(f.ty * scale, MAX_TORQUE_NM),
        z: clamp(f.tz * scale, MAX_TORQUE_NM),
      },
      true,
    );
  };

  if (bodies instanceof Map) {
    for (const [id, b] of bodies) visit(id, b);
  } else {
    const arr = bodies as ReadonlyArray<RapierBodyLike | null | undefined>;
    for (let i = 1; i < arr.length; i++) {
      const b = arr[i];
      if (b) visit(i, b);
    }
  }
}

// ---------------------------------------------------------------------------
// Stabilized force application (CPU buoyancy + EMA smoothing + velocity clamp)
// ---------------------------------------------------------------------------

export interface ForceSmootherState {
  prev: Map<number, ChunkForce>;
}

export function createForceSmootherState(): ForceSmootherState {
  return { prev: new Map() };
}

/** EMA-smooth GPU forces (horizontal only; fy and torques zeroed since
 *  fy uses CPU buoyancy and torques from stale positions cause violent spinning). */
function smoothGpuForce(raw: ChunkForce, prev: ChunkForce | undefined, alpha: number): ChunkForce {
  if (!prev) return { fx: raw.fx, fy: 0, fz: raw.fz, tx: 0, ty: 0, tz: 0 };
  return {
    fx: alpha * raw.fx + (1 - alpha) * prev.fx,
    fy: 0,
    fz: alpha * raw.fz + (1 - alpha) * prev.fz,
    tx: 0,
    ty: 0,
    tz: 0,
  };
}

export interface CoupledBodyInfo {
  chunkId: number;
  body: RapierBodyFull;
  halfExtents: [number, number, number];
  waterLevelRef: number;
  bedLevelRef: number;
}

export interface StabilizedForcesOptions {
  alpha?: number;        // EMA smoothing factor (default 0.7)
  maxVy?: number;        // vertical velocity clamp (default 3 m/s)
  maxHorizontal?: number; // horizontal velocity clamp (default 5 m/s)
  /** Per-frame vertical velocity damping (0..1, default 0.85).
   *  Applied after Rapier step to suppress oscillation from
   *  rasterizer↔buoyancy feedback. */
  vyDamping?: number;
}

export function applyStabilizedForces(
  bodies: ReadonlyMap<number, CoupledBodyInfo>,
  gpuForces: ReadonlyArray<ChunkForce> | null,
  smoother: ForceSmootherState,
  dx: number,
  opts: StabilizedForcesOptions = {},
): void {
  const alpha = opts.alpha ?? 0.7;
  const maxVy = opts.maxVy ?? 3.0;
  const maxH = opts.maxHorizontal ?? 5.0;

  for (const [_key, cb] of bodies) {
    const { chunkId, body, halfExtents, waterLevelRef, bedLevelRef } = cb;
    const [halfX, halfY, halfZ] = halfExtents;

    // 1) Reset forces
    body.resetForces(true);
    body.resetTorques(true);

    // 2) CPU buoyancy (immediate, from current position)
    const vel = body.linvel();
    const footprint = estimateFootprintCells(halfX, halfZ, dx);
    const comY = body.translation().y;
    const buoyancy = computeCpuBuoyancy({
      comY,
      halfY,
      linvelY: vel.y,
      waterLevel: waterLevelRef,
      bedLevel: bedLevelRef,
      footprintCells: footprint,
      dx,
    });
    const vdamp = computeVerticalDamping(vel.y, buoyancy.submergedFraction, dx, footprint);

    // 3) Smooth GPU drag/pressure forces
    let smoothed: ChunkForce = { fx: 0, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 };
    if (gpuForces) {
      const raw = gpuForces[chunkId];
      if (raw) {
        smoothed = smoothGpuForce(raw, smoother.prev.get(chunkId), alpha);
        smoother.prev.set(chunkId, smoothed);
      }
    }

    // 4) Apply combined forces
    const fy = buoyancy.fy + vdamp;
    body.addForce(
      {
        x: clamp(smoothed.fx, MAX_FORCE_NEWTONS),
        y: clamp(fy, MAX_FORCE_NEWTONS),
        z: clamp(smoothed.fz, MAX_FORCE_NEWTONS),
      },
      true,
    );
    body.addTorque(
      {
        x: clamp(smoothed.tx, MAX_TORQUE_NM),
        y: clamp(smoothed.ty, MAX_TORQUE_NM),
        z: clamp(smoothed.tz, MAX_TORQUE_NM),
      },
      true,
    );

  }
}

/**
 * Clamp coupled body velocities after Rapier step (safety net).
 * Must be called AFTER `world.step()` to be effective.
 *
 * `vyDamping` and `maxVy` are stability tools for the buoyancy↔rasterizer
 * feedback loop — applying them to a body that's still in free-fall (well
 * above its reference water surface) caps it at terminal ~0.67 m/s and
 * the body appears to hover. We gate both on whether the body bottom is
 * within `AIR_MARGIN` of the bed+water reference height; outside that
 * margin the body is treated as in air and its vertical velocity is left
 * alone. Horizontal clamping still applies (cheap, never bites in air).
 */
const AIR_MARGIN_METRES = 0.1;

export function clampCoupledVelocities(
  bodies: ReadonlyMap<number, CoupledBodyInfo>,
  opts: StabilizedForcesOptions = {},
): void {
  const maxVy = opts.maxVy ?? 3.0;
  const maxH = opts.maxHorizontal ?? 5.0;
  const vyDamp = opts.vyDamping ?? 0.85;
  for (const [_key, cb] of bodies) {
    const lv = cb.body.linvel();
    const halfY = cb.halfExtents[1] ?? 0;
    const t = cb.body.translation();
    const bodyBottom = t.y - halfY;
    const surfaceY = cb.bedLevelRef + cb.waterLevelRef;
    const inAir = bodyBottom > surfaceY + AIR_MARGIN_METRES;

    const cx = clamp(lv.x, maxH);
    const cz = clamp(lv.z, maxH);
    // In air: leave vy alone. In/near water: dampen vertical velocity each
    // frame to suppress rasterizer↔buoyancy oscillation, then clamp.
    const cy = inAir ? lv.y : clamp(lv.y * vyDamp, maxVy);
    if (lv.x !== cx || lv.y !== cy || lv.z !== cz) {
      cb.body.setLinvel({ x: cx, y: cy, z: cz }, true);
    }
  }
}
