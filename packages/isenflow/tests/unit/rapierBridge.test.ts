import { describe, it, expect, vi } from 'vitest';
import {
  applyForcesToBodies,
  applyStabilizedForces,
  clampCoupledVelocities,
  createForceSmootherState,
  type RapierBodyFull,
  type CoupledBodyInfo,
} from '../../src/coupling/RapierBridge.js';
import type { ChunkForce } from '../../src/coupling/ForceReadback.js';

/** Create a mock Rapier body with tracked state. */
function mockBody(opts?: {
  pos?: { x: number; y: number; z: number };
  vel?: { x: number; y: number; z: number };
}): RapierBodyFull & {
  _force: { x: number; y: number; z: number };
  _torque: { x: number; y: number; z: number };
  _vel: { x: number; y: number; z: number };
  _pos: { x: number; y: number; z: number };
} {
  const pos = { x: 0, y: 0.8, z: 0, ...opts?.pos };
  const vel = { x: 0, y: 0, z: 0, ...opts?.vel };
  const force = { x: 0, y: 0, z: 0 };
  const torque = { x: 0, y: 0, z: 0 };
  return {
    _force: force,
    _torque: torque,
    _vel: vel,
    _pos: pos,
    resetForces: vi.fn(() => { force.x = 0; force.y = 0; force.z = 0; }),
    resetTorques: vi.fn(() => { torque.x = 0; torque.y = 0; torque.z = 0; }),
    addForce: vi.fn((f: { x: number; y: number; z: number }) => {
      force.x += f.x; force.y += f.y; force.z += f.z;
    }),
    addTorque: vi.fn((t: { x: number; y: number; z: number }) => {
      torque.x += t.x; torque.y += t.y; torque.z += t.z;
    }),
    linvel: () => ({ ...vel }),
    setLinvel: vi.fn((v: { x: number; y: number; z: number }) => {
      vel.x = v.x; vel.y = v.y; vel.z = v.z;
    }),
    translation: () => ({ ...pos }),
  };
}

function zeroForce(): ChunkForce {
  return { fx: 0, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 };
}

describe('RapierBridge', () => {
  // ---- applyForcesToBodies ----

  describe('applyForcesToBodies', () => {
    it('applies forces from Map-based body lookup', () => {
      const body = mockBody();
      const forces: ChunkForce[] = [zeroForce(), { fx: 100, fy: 50, fz: 0, tx: 0, ty: 0, tz: 0 }];
      const map = new Map([[1, body]]);
      applyForcesToBodies(map, forces);
      expect(body._force.x).toBeCloseTo(100);
      expect(body._force.y).toBeCloseTo(50);
    });

    it('applies forces from array-based body lookup', () => {
      const body = mockBody();
      const forces: ChunkForce[] = [zeroForce(), { fx: 200, fy: 0, fz: -30, tx: 0, ty: 0, tz: 0 }];
      applyForcesToBodies([null, body], forces);
      expect(body._force.x).toBeCloseTo(200);
      expect(body._force.z).toBeCloseTo(-30);
    });

    it('clamps extreme forces to MAX_FORCE (5000 N)', () => {
      const body = mockBody();
      const forces: ChunkForce[] = [zeroForce(), { fx: 999999, fy: -999999, fz: 0, tx: 0, ty: 0, tz: 0 }];
      applyForcesToBodies(new Map([[1, body]]), forces);
      expect(body._force.x).toBe(5000);
      expect(body._force.y).toBe(-5000);
    });

    it('clamps NaN/Infinity to zero', () => {
      const body = mockBody();
      const forces: ChunkForce[] = [zeroForce(), { fx: NaN, fy: Infinity, fz: -Infinity, tx: 0, ty: 0, tz: 0 }];
      applyForcesToBodies(new Map([[1, body]]), forces);
      expect(body._force.x).toBe(0);
      expect(body._force.y).toBe(0);
      expect(body._force.z).toBe(0);
    });

    it('skips chunk id 0', () => {
      const body = mockBody();
      const forces: ChunkForce[] = [{ fx: 100, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 }];
      applyForcesToBodies(new Map([[0, body]]), forces);
      expect(body.addForce).not.toHaveBeenCalled();
    });

    it('applies scale factor', () => {
      const body = mockBody();
      const forces: ChunkForce[] = [zeroForce(), { fx: 100, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 }];
      applyForcesToBodies(new Map([[1, body]]), forces, { scale: 0.5 });
      expect(body._force.x).toBeCloseTo(50);
    });
  });

  // ---- EMA smoothing (tested via applyStabilizedForces) ----

  describe('applyStabilizedForces — EMA smoothing', () => {
    const dx = 0.125;

    function makeCoupledBody(body: RapierBodyFull): CoupledBodyInfo {
      return {
        chunkId: 1,
        body,
        halfExtents: [0.2, 0.1, 0.2],
        waterLevelRef: 0.8,
        bedLevelRef: 0,
      };
    }

    it('first frame uses raw GPU force (no prior smoothing)', () => {
      const body = mockBody({ pos: { x: 0, y: 0.8, z: 0 } });
      const smoother = createForceSmootherState();
      const forces: ChunkForce[] = [zeroForce(), { fx: 1000, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 }];
      const bodies = new Map([[1, makeCoupledBody(body)]]);
      applyStabilizedForces(bodies, forces, smoother, dx);
      // First frame: smoothed = raw (with fy zeroed). fx should be applied.
      expect(body._force.x).toBeCloseTo(1000);
    });

    it('subsequent frames EMA-smooth toward new values', () => {
      const body = mockBody({ pos: { x: 0, y: 0.8, z: 0 } });
      const smoother = createForceSmootherState();
      const bodies = new Map([[1, makeCoupledBody(body)]]);
      const alpha = 0.7;

      // Frame 1: raw = 1000
      const f1: ChunkForce[] = [zeroForce(), { fx: 1000, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 }];
      applyStabilizedForces(bodies, f1, smoother, dx, { alpha });
      const fx1 = body._force.x;

      // Frame 2: raw = 0 (sudden drop)
      const f2: ChunkForce[] = [zeroForce(), { fx: 0, fy: 0, fz: 0, tx: 0, ty: 0, tz: 0 }];
      applyStabilizedForces(bodies, f2, smoother, dx, { alpha });
      const fx2 = body._force.x;

      // EMA: smoothed = alpha*0 + (1-alpha)*1000 = 300
      // Force should be about 300 (may be clamped if > 5000)
      expect(fx2).toBeCloseTo(300, 0);
      expect(fx2).toBeLessThan(fx1); // Smoothing reduces the spike
    });

    it('GPU fy and torques are zeroed (stale data causes instability)', () => {
      const body = mockBody({ pos: { x: 0, y: 0.8, z: 0 } });
      const smoother = createForceSmootherState();
      const forces: ChunkForce[] = [zeroForce(), { fx: 0, fy: 9999, fz: 0, tx: 5000, ty: 3000, tz: 4000 }];
      const bodies = new Map([[1, makeCoupledBody(body)]]);
      applyStabilizedForces(bodies, forces, smoother, dx);
      // fy from GPU should be zeroed; only CPU buoyancy contributes to fy
      expect(Math.abs(body._force.y)).toBeLessThan(1000);
      // Torques from GPU should be zeroed (stale positions cause violent spinning)
      expect(body._torque.x).toBe(0);
      expect(body._torque.y).toBe(0);
      expect(body._torque.z).toBe(0);
    });

    it('handles null GPU forces gracefully', () => {
      const body = mockBody({ pos: { x: 0, y: 0.8, z: 0 } });
      const smoother = createForceSmootherState();
      const bodies = new Map([[1, makeCoupledBody(body)]]);
      applyStabilizedForces(bodies, null, smoother, dx);
      // Should not throw; only CPU buoyancy should be applied
      expect(body.addForce).toHaveBeenCalled();
    });
  });

  // ---- clampCoupledVelocities ----

  describe('clampCoupledVelocities', () => {
    it('clamps horizontal velocity to maxHorizontal', () => {
      const body = mockBody({ vel: { x: 20, y: 0, z: -15 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.2, 0.1, 0.2] as [number, number, number],
        waterLevelRef: 0.8, bedLevelRef: 0,
      }]]);
      clampCoupledVelocities(bodies, { maxHorizontal: 5 });
      expect(body._vel.x).toBe(5);
      expect(body._vel.z).toBe(-5);
    });

    it('clamps vertical velocity to maxVy', () => {
      const body = mockBody({ vel: { x: 0, y: 10, z: 0 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.2, 0.1, 0.2] as [number, number, number],
        waterLevelRef: 0.8, bedLevelRef: 0,
      }]]);
      clampCoupledVelocities(bodies, { maxVy: 3, vyDamping: 1.0 });
      expect(body._vel.y).toBe(3);
    });

    it('applies vertical velocity damping', () => {
      const body = mockBody({ vel: { x: 0, y: 2.0, z: 0 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.2, 0.1, 0.2] as [number, number, number],
        waterLevelRef: 0.8, bedLevelRef: 0,
      }]]);
      clampCoupledVelocities(bodies, { vyDamping: 0.5 });
      expect(body._vel.y).toBeCloseTo(1.0); // 2.0 * 0.5
    });

    it('does not modify velocity within limits (with damping applied)', () => {
      // vy=0 is unaffected by damping, and x/z are within limits
      const body = mockBody({ vel: { x: 1, y: 0, z: 2 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.2, 0.1, 0.2] as [number, number, number],
        waterLevelRef: 0.8, bedLevelRef: 0,
      }]]);
      clampCoupledVelocities(bodies);
      expect(body.setLinvel).not.toHaveBeenCalled();
    });

    it('uses default limits (maxVy=3, maxH=5, vyDamping=0.85)', () => {
      const body = mockBody({ vel: { x: 100, y: -100, z: 0 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.2, 0.1, 0.2] as [number, number, number],
        waterLevelRef: 0.8, bedLevelRef: 0,
      }]]);
      clampCoupledVelocities(bodies);
      expect(body._vel.x).toBe(5);
      // -100 * 0.85 = -85, clamped to -3
      expect(body._vel.y).toBe(-3);
    });

    it('leaves vy alone for bodies in free fall (well above water surface)', () => {
      // Body at y=20, halfY=0.25. Reference surface = bed(0) + water(1.5) = 1.5.
      // Body bottom = 19.75 ≫ 1.5 + 0.1 → in air → no damping, no clamp.
      const body = mockBody({ pos: { x: 0, y: 20, z: 0 }, vel: { x: 0, y: -50, z: 0 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.25, 0.25, 0.25] as [number, number, number],
        waterLevelRef: 1.5, bedLevelRef: 0,
      }]]);
      clampCoupledVelocities(bodies);
      // setLinvel was not called because (x=0, z=0) and y is unchanged.
      expect(body.setLinvel).not.toHaveBeenCalled();
      expect(body._vel.y).toBe(-50);
    });

    it('damps + clamps vy when the body bottom is within the in-water margin', () => {
      // Body at y=0.7 with halfY=0.1 → bottom=0.6. Surface=0.8. 0.6 ≤ 0.9
      // → in water → vy damped & clamped.
      const body = mockBody({ pos: { x: 0, y: 0.7, z: 0 }, vel: { x: 0, y: -100, z: 0 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.2, 0.1, 0.2] as [number, number, number],
        waterLevelRef: 0.8, bedLevelRef: 0,
      }]]);
      clampCoupledVelocities(bodies);
      expect(body._vel.y).toBe(-3); // -100 * 0.85 = -85, clamped to -3
    });

    it('still clamps horizontal velocity even when the body is in air', () => {
      // High-y body with large horizontal velocity. Horizontal should still
      // be clamped (cheap, no behaviour change in air); only vy is freed.
      const body = mockBody({ pos: { x: 0, y: 20, z: 0 }, vel: { x: 20, y: -10, z: -15 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.25, 0.25, 0.25] as [number, number, number],
        waterLevelRef: 1.5, bedLevelRef: 0,
      }]]);
      clampCoupledVelocities(bodies);
      expect(body._vel.x).toBe(5);
      expect(body._vel.z).toBe(-5);
      expect(body._vel.y).toBe(-10); // untouched
    });

    it('bedLevelRef > 0 shifts the water-surface threshold', () => {
      // Elevated water (bed=2, water=1) → surface=3. Body at y=10 is in air.
      const body = mockBody({ pos: { x: 0, y: 10, z: 0 }, vel: { x: 0, y: -30, z: 0 } });
      const bodies = new Map([[1, {
        chunkId: 1, body,
        halfExtents: [0.2, 0.2, 0.2] as [number, number, number],
        waterLevelRef: 1.0, bedLevelRef: 2.0,
      }]]);
      clampCoupledVelocities(bodies);
      expect(body._vel.y).toBe(-30);
    });
  });

  // ---- Force cap regression test ----

  describe('force capping regression', () => {
    it('forces above 5000 N are clamped (prevents runaway bodies)', () => {
      const body = mockBody();
      const hugeForce: ChunkForce = { fx: 180000, fy: 0, fz: -120000, tx: 50000, ty: 0, tz: 0 };
      applyForcesToBodies(new Map([[1, body]]), [zeroForce(), hugeForce]);
      expect(body._force.x).toBe(5000);
      expect(body._force.z).toBe(-5000);
      expect(body._torque.x).toBe(5000);
    });
  });
});
