/**
 * Applies decoded ChunkForce values to Rapier rigid bodies.
 *
 * `@dimforge/rapier3d-compat` is a peer dependency: we type-only-import here
 * to avoid the dependency in pure-math test runs.
 */
import type { ChunkForce } from './ForceReadback.js';

/** Minimal duck type of `RigidBody` we depend on. */
export interface RapierBodyLike {
  resetForces(wakeUp: boolean): void;
  resetTorques(wakeUp: boolean): void;
  addForce(force: { x: number; y: number; z: number }, wakeUp: boolean): void;
  addTorque(torque: { x: number; y: number; z: number }, wakeUp: boolean): void;
}

/** Map keyed by chunk id (0 reserved). */
export type RapierBodyMap = ReadonlyMap<number, RapierBodyLike>;

/** Sanity bound on a single per-frame force contribution. */
const MAX_FORCE_NEWTONS = 5e6;
const MAX_TORQUE_NM = 5e6;

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
