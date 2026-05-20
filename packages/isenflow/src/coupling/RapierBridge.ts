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

/**
 * `bodies` indexed by chunk id (skip slot 0 which is "no chunk").
 */
export function applyForcesToBodies(
  bodies: ReadonlyArray<RapierBodyLike | null | undefined>,
  forces: ReadonlyArray<ChunkForce>,
): void {
  const n = Math.min(bodies.length, forces.length);
  for (let i = 1; i < n; i++) {
    const body = bodies[i];
    const f = forces[i];
    if (!body || !f) continue;
    body.resetForces(true);
    body.resetTorques(true);
    body.addForce({ x: f.fx, y: f.fy, z: f.fz }, true);
    body.addTorque({ x: f.tx, y: f.ty, z: f.tz }, true);
  }
}
