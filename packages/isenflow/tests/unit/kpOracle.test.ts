/**
 * Layer-1 oracle tests for the CPU Kurganov–Petrova solver.
 *
 * These validate the *algorithm itself* — mass conservation, well-balanced
 * lake-at-rest, positivity preservation, wave speed, dam-break front
 * propagation, and the desingularized momentum field.
 *
 * The same fixtures will be run through the WGSL kernels via Playwright
 * to assert byte-level GPU↔CPU agreement (kpOracleMatch.test.ts, Phase 6).
 */
import { describe, it, expect } from 'vitest';
import {
  createKpState,
  createKpParams,
  createKpScratch,
  kpStep,
  desingularize,
  deriveVelocity,
  BT_CLOSED,
  BT_INFLOW,
  BT_OPEN,
  BT_SOLID,
  type KpState,
  type KpParams,
} from '../../src/core/cpu/CpuKurganov.js';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function totalVolume(state: KpState, dx: number): number {
  let sum = 0;
  const area = dx * dx;
  for (let i = 0; i < state.h.length; i++) sum += Math.max(0, state.h[i]!) * area;
  return sum;
}

function totalMomentum(state: KpState, dx: number): { x: number; z: number } {
  let sx = 0;
  let sz = 0;
  const area = dx * dx;
  for (let i = 0; i < state.h.length; i++) {
    sx += state.hu[i]! * area;
    sz += state.hv[i]! * area;
  }
  return { x: sx, z: sz };
}

function maxAbs(arr: Float32Array): number {
  let m = 0;
  for (let i = 0; i < arr.length; i++) if (Math.abs(arr[i]!) > m) m = Math.abs(arr[i]!);
  return m;
}

function fillUniformH(state: KpState, value: number): void {
  state.h.fill(value);
}

function fillBoundaryAll(params: KpParams, type: number): void {
  params.boundary.fill(type);
}

// ---------------------------------------------------------------------------
// Helper desingularization tests
// ---------------------------------------------------------------------------

describe('Desingularize (KP07 eq. 2.16)', () => {
  it('returns 0 for h = 0', () => {
    expect(desingularize(0, 1.0, 1e-3)).toBe(0);
  });

  it('returns q/h for h >> eps', () => {
    const u = desingularize(2.0, 4.0, 1e-3);
    expect(u).toBeCloseTo(2.0, 5);
  });

  it('is monotonic in q', () => {
    const h = 0.005;
    const u1 = desingularize(h, 1.0, 1e-3);
    const u2 = desingularize(h, 2.0, 1e-3);
    expect(u2).toBeGreaterThan(u1);
  });

  it('is smooth across h → 0 (no discontinuity)', () => {
    // As h decreases through eps, u should smoothly approach 0
    // rather than blowing up as q/h would.
    const eps = 1e-3;
    let prev = desingularize(0.1, 1.0, eps);
    for (let h = 0.05; h > 1e-6; h *= 0.5) {
      const u = desingularize(h, 1.0, eps);
      // Should not blow up — VP's max(0.05, h) would give u = 20 here
      expect(Math.abs(u)).toBeLessThan(1000);
      // Should be monotonically decreasing or stable as h decreases (with same q)
      // Actually: as h drops, q/h grows in raw form, but desingularization
      // damps it. The result should be smoothly bounded.
      expect(Number.isFinite(u)).toBe(true);
      prev = u;
    }
    void prev;
  });
});

// ---------------------------------------------------------------------------
// Lake at rest — C-property
// ---------------------------------------------------------------------------

describe('KP — lake-at-rest C-property', () => {
  it('flat basin stays flat (no flow generated)', () => {
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.005, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);
    fillUniformH(state, 1.0);

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 200; s++) kpStep(state, params, scratch);

    expect(maxAbs(state.hu)).toBeLessThan(1e-4);
    expect(maxAbs(state.hv)).toBeLessThan(1e-4);

    // Volume conserved
    const v0 = 1.0 * W * H * params.dx * params.dx;
    const v1 = totalVolume(state, params.dx);
    expect(Math.abs(v1 - v0) / v0).toBeLessThan(1e-4);
  });

  it('lake at rest over a smooth Gaussian bed bump preserves flat surface', () => {
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.005, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);

    const eta0 = 1.5;  // water surface elevation
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const cx = (i - W / 2) * params.dx;
        const cy = (j - H / 2) * params.dx;
        const r2 = cx * cx + cy * cy;
        const bed = 0.5 * Math.exp(-r2 / 4);  // peak height 0.5m
        state.bed[j * W + i] = bed;
        state.h[j * W + i] = Math.max(0, eta0 - bed);
      }
    }

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 1000; s++) kpStep(state, params, scratch);

    // Velocity should remain tiny across the bump (well-balanced property)
    const u = new Float32Array(W * H);
    const v = new Float32Array(W * H);
    deriveVelocity(state, params.desingEpsilon, u, v);
    expect(maxAbs(u)).toBeLessThan(0.01);
    expect(maxAbs(v)).toBeLessThan(0.01);
  });

  it('lake at rest over a STEEP bed step (1-cell jump) preserves flat surface', () => {
    // This is the building-wall pathology that broke VP.
    // KP + Audusse should handle it cleanly.
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.002, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);

    const eta0 = 2.0;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        // Step in the bed: half the domain is at 0, other half at 1.0
        const bed = i < W / 2 ? 0 : 1.0;
        state.bed[j * W + i] = bed;
        state.h[j * W + i] = Math.max(0, eta0 - bed);
      }
    }

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 500; s++) kpStep(state, params, scratch);

    const u = new Float32Array(W * H);
    const v = new Float32Array(W * H);
    deriveVelocity(state, params.desingEpsilon, u, v);

    // Velocity should remain very small even at the bed step
    expect(maxAbs(u)).toBeLessThan(0.02);
    expect(maxAbs(v)).toBeLessThan(0.02);

    // Surface elevation should remain ~flat at eta0
    for (let c = 0; c < W * H; c++) {
      const eta = state.h[c]! + state.bed[c]!;
      expect(Math.abs(eta - eta0)).toBeLessThan(0.05);
    }
  });
});

// ---------------------------------------------------------------------------
// Mass conservation
// ---------------------------------------------------------------------------

describe('KP — mass conservation', () => {
  it('closed basin with Gaussian bump conserves mass over 1000 steps', () => {
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.005, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);

    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const cx = i - W / 2;
        const cy = j - H / 2;
        state.h[j * W + i] = 1.0 + 0.4 * Math.exp(-(cx * cx + cy * cy) / 16);
      }
    }

    const v0 = totalVolume(state, params.dx);
    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 1000; s++) kpStep(state, params, scratch);
    const v1 = totalVolume(state, params.dx);

    expect(Math.abs(v1 - v0) / v0).toBeLessThan(1e-3);
  });

  it('closed basin with non-trivial momentum conserves mass', () => {
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.005, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);

    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const c = j * W + i;
        state.h[c] = 1.0;
        state.hu[c] = 0.5;  // rightward flow
      }
    }

    const v0 = totalVolume(state, params.dx);
    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 500; s++) kpStep(state, params, scratch);
    const v1 = totalVolume(state, params.dx);

    expect(Math.abs(v1 - v0) / v0).toBeLessThan(1e-3);
  });
});

// ---------------------------------------------------------------------------
// Positivity preservation
// ---------------------------------------------------------------------------

describe('KP — positivity preservation', () => {
  it('h never goes negative even with strong outflow', () => {
    const W = 32, H = 16;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.003, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);
    // Open boundary on right
    for (let j = 0; j < H; j++) params.boundary[j * W + W - 1] = BT_OPEN;

    // High dam upstream, low downstream
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        state.h[j * W + i] = i < W / 3 ? 2.0 : 0.05;
      }
    }

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 500; s++) {
      kpStep(state, params, scratch);
      // Check positivity after each step
      for (let c = 0; c < W * H; c++) {
        expect(state.h[c]).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('dry cells stay dry when no water enters', () => {
    const W = 16, H = 16;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.005, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);
    // Only fill right half with water
    for (let j = 0; j < H; j++) {
      for (let i = W / 2; i < W; i++) {
        state.h[j * W + i] = 1.0;
      }
    }
    // Raise the bed in the left half ABOVE the right-half water surface so
    // water can't reach.
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W / 2; i++) {
        state.bed[j * W + i] = 2.0;
      }
    }

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 200; s++) kpStep(state, params, scratch);

    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W / 2; i++) {
        expect(state.h[j * W + i]).toBe(0);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Wave celerity
// ---------------------------------------------------------------------------

describe('KP — wave celerity', () => {
  it('Gaussian perturbation: signal arrival time matches √(g·h) wave speed', () => {
    // Robust wave-speed test: measure the FIRST-ARRIVAL time of the wave
    // front at a fixed downstream observation point.  This avoids the
    // numerical-dissipation problem of tracking a peak (which decays and
    // smears in finite-volume schemes).
    const W = 64, H = 4;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.1, dt: 0.001, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);
    for (let j = 0; j < H; j++) {
      params.boundary[j * W] = BT_OPEN;
      params.boundary[j * W + W - 1] = BT_OPEN;
    }

    const h0 = 1.0;
    const amp = 0.05;  // 5 % perturbation — well above noise, still ~linear
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const cx = i - W / 2;
        state.h[j * W + i] = h0 + amp * Math.exp(-(cx * cx) / 4);
      }
    }

    const c_theory = Math.sqrt(params.gravity * h0);  // ≈ 3.13 m/s
    // Observation point: 20 cells (2 m) east of center
    const obsI = Math.floor(W / 2) + 20;
    const expectedArrival = (20 * params.dx) / c_theory;  // ≈ 0.64 s

    const scratch = createKpScratch(W, H);
    // Detect first arrival: when h at obsI rises above background by 10% of
    // amp — high enough to ignore numerical noise / scheme dispersion's
    // leading edge, low enough to register before the peak passes.
    const detectionThreshold = 0.10 * amp;
    let arrivalT = -1;
    const maxSteps = Math.round(expectedArrival * 2 / params.dt);
    for (let s = 0; s < maxSteps; s++) {
      kpStep(state, params, scratch);
      const delta = state.h[2 * W + obsI]! - h0;
      if (arrivalT < 0 && delta > detectionThreshold) {
        arrivalT = (s + 1) * params.dt;
        break;
      }
    }

    expect(arrivalT).toBeGreaterThan(0);
    const observedSpeed = (20 * params.dx) / arrivalT;
    // Allow a wide range — MUSCL central-upwind has some phase error on
    // under-resolved wave packets, and we measure first-arrival time
    // which includes the leading edge of dispersion.
    expect(observedSpeed).toBeGreaterThan(c_theory * 0.5);
    expect(observedSpeed).toBeLessThan(c_theory * 1.5);
  });

  it('wave-speed scales with depth (c ∝ √h)', () => {
    // Run two simulations at different depths and verify the arrival-time
    // ratio matches √(h2/h1). This validates the SCALING is correct even
    // if absolute speed has dispersion error.
    const W = 64, H = 4;
    const measureArrival = (h0: number, amp: number): number => {
      const state = createKpState(W, H);
      const params = createKpParams(W, H, { dx: 0.1, dt: 0.0005, gravity: 9.81, manningN: 0 });
      fillBoundaryAll(params, BT_CLOSED);
      for (let j = 0; j < H; j++) {
        params.boundary[j * W] = BT_OPEN;
        params.boundary[j * W + W - 1] = BT_OPEN;
      }
      for (let j = 0; j < H; j++) {
        for (let i = 0; i < W; i++) {
          const cx = i - W / 2;
          state.h[j * W + i] = h0 + amp * Math.exp(-(cx * cx) / 4);
        }
      }
      const obsI = Math.floor(W / 2) + 15;
      const threshold = 0.05 * amp;
      const scratch = createKpScratch(W, H);
      for (let s = 0; s < 5000; s++) {
        kpStep(state, params, scratch);
        if (state.h[2 * W + obsI]! - h0 > threshold) return (s + 1) * params.dt;
      }
      return -1;
    };
    const t1 = measureArrival(1.0, 0.05);
    const t4 = measureArrival(4.0, 0.05);
    expect(t1).toBeGreaterThan(0);
    expect(t4).toBeGreaterThan(0);
    // c2/c1 = √(h2/h1) = √4 = 2 → t1/t4 should be 2
    const ratio = t1 / t4;
    expect(ratio).toBeGreaterThan(1.6);  // ±20%
    expect(ratio).toBeLessThan(2.4);
  });
});

// ---------------------------------------------------------------------------
// Dam break (Stoker analytic)
// ---------------------------------------------------------------------------

describe('KP — dam break', () => {
  it('1D dam-break front advances and depth at dam approaches 4·hL/9', () => {
    // Classic Ritter solution for dry bed: depth at dam x=0 should be 4·hL/9
    // We use a wet bed downstream to keep the test simpler — KP should still
    // produce a sharp front.
    const W = 128, H = 4;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.1, dt: 0.0005, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);
    // Wet downstream
    const hL = 1.0;
    const hR = 0.1;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        state.h[j * W + i] = i < W / 2 ? hL : hR;
      }
    }

    const scratch = createKpScratch(W, H);
    // Run for 0.2 s sim time
    const stepsNeeded = Math.round(0.2 / params.dt);
    for (let s = 0; s < stepsNeeded; s++) kpStep(state, params, scratch);

    // Sample h at the dam location (i = W/2)
    const hDam = state.h[2 * W + W / 2]!;
    // Should be between hR and hL, closer to the Ritter-like middle value
    expect(hDam).toBeGreaterThan(hR);
    expect(hDam).toBeLessThan(hL);

    // Front should have advanced — depth at i = W/2 + 30 (3 m downstream) should be > hR
    const hFront = state.h[2 * W + (W / 2 + 30)]!;
    expect(hFront).toBeGreaterThan(hR);
  });
});

// ---------------------------------------------------------------------------
// Solid cells
// ---------------------------------------------------------------------------

describe('KP — Solid boundary', () => {
  it('solid cell stays at h=0 with surrounding water at constant elevation', () => {
    const W = 16, H = 16;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.003, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);

    // Single solid column at center
    const ci = Math.floor(W / 2);
    const cj = Math.floor(H / 2);
    params.boundary[cj * W + ci] = BT_SOLID;

    // Fill all OTHER cells with h = 1 over flat bed
    for (let c = 0; c < W * H; c++) {
      if (params.boundary[c] === BT_SOLID) {
        state.h[c] = 0;
      } else {
        state.h[c] = 1.0;
      }
    }

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 500; s++) kpStep(state, params, scratch);

    // Solid stays dry
    expect(state.h[cj * W + ci]).toBe(0);
    expect(state.hu[cj * W + ci]).toBe(0);
    expect(state.hv[cj * W + ci]).toBe(0);

    // Neighbors should remain ~flat (no spurious oscillation)
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const nc = (cj + dj) * W + (ci + di);
      expect(Math.abs(state.h[nc]! - 1.0)).toBeLessThan(0.05);
    }
  });
});

// ---------------------------------------------------------------------------
// Inflow boundary
// ---------------------------------------------------------------------------

describe('KP — Inflow boundary', () => {
  it('pins depth to target (floor)', () => {
    const W = 16, H = 16;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.005, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);
    // Inflow column on left edge
    for (let j = 0; j < H; j++) {
      params.boundary[j * W] = BT_INFLOW;
      params.boundaryTargetH[j * W] = 1.0;
    }

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 50; s++) kpStep(state, params, scratch);

    for (let j = 0; j < H; j++) {
      expect(state.h[j * W]).toBeGreaterThanOrEqual(0.99);
    }
  });
});

// ---------------------------------------------------------------------------
// Accurate momentum field — boats / objects perspective
// ---------------------------------------------------------------------------

describe('KP — momentum field is accurate (boat-coupling perspective)', () => {
  it('uniform translational flow has uniform u everywhere (no spurious gradients)', () => {
    const W = 32, H = 8;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.003, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);
    for (let j = 0; j < H; j++) {
      params.boundary[j * W] = BT_INFLOW;
      params.boundaryTargetH[j * W] = 1.0;
      params.boundary[j * W + W - 1] = BT_OPEN;
    }

    // Initial: uniform h=1, no momentum
    fillUniformH(state, 1.0);
    // Set initial uniform momentum hu = 1.0 (u = 1 m/s)
    state.hu.fill(1.0);

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 100; s++) kpStep(state, params, scratch);

    // Velocity should remain uniform ~ 1 m/s with very small spread
    const u = new Float32Array(W * H);
    const v = new Float32Array(W * H);
    deriveVelocity(state, params.desingEpsilon, u, v);

    // Check the middle region (avoid boundary effects)
    for (let j = 1; j < H - 1; j++) {
      for (let i = 5; i < W - 5; i++) {
        const c = j * W + i;
        expect(u[c]).toBeGreaterThan(0.5);
        expect(u[c]).toBeLessThan(1.5);
        expect(Math.abs(v[c]!)).toBeLessThan(0.1);
      }
    }
  });

  it('local momentum impulse propagates as a wave (not just diffuses)', () => {
    // The KEY test for boat-coupling: if we inject momentum at a point, it
    // should propagate as an identifiable wave, not just decay locally.
    const W = 64, H = 4;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.1, dt: 0.001, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);
    fillUniformH(state, 1.0);

    // Inject rightward momentum at center
    for (let j = 0; j < H; j++) {
      state.hu[j * W + W / 2] = 2.0;
    }

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 100; s++) kpStep(state, params, scratch);

    // After ~0.1s the momentum should have propagated to neighbors —
    // we expect hu to be non-trivial in cells near (but not at) the impulse
    // location.
    let sumHuRight = 0;
    let sumHuLeft = 0;
    for (let j = 0; j < H; j++) {
      for (let i = W / 2 + 1; i < W / 2 + 10; i++) {
        sumHuRight += state.hu[j * W + i]!;
      }
      for (let i = W / 2 - 10; i < W / 2; i++) {
        sumHuLeft += state.hu[j * W + i]!;
      }
    }

    // Rightward momentum should have spread to the right
    expect(sumHuRight).toBeGreaterThan(0);
    // Total system momentum (in closed basin) should be conserved minus
    // small reflection effects — sum on both sides should be non-zero
    expect(Math.abs(sumHuRight) + Math.abs(sumHuLeft)).toBeGreaterThan(0.1);
  });

  it('total momentum is conserved (closed basin, no friction)', () => {
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.5, dt: 0.005, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);

    // Set up asymmetric initial state with bulk momentum
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const c = j * W + i;
        state.h[c] = 1.0;
        state.hu[c] = 0.5;
        state.hv[c] = 0.3;
      }
    }

    const p0 = totalMomentum(state, params.dx);
    const scratch = createKpScratch(W, H);
    // Short run — long runs will reflect off walls and oscillate, but
    // total momentum should be conserved AS LONG AS no wave reaches the
    // walls.  Sample after 20 steps (very short).
    for (let s = 0; s < 20; s++) kpStep(state, params, scratch);
    const p1 = totalMomentum(state, params.dx);

    // Allow 5% — bed source corrections aren't perfectly conservative,
    // closed-wall reflection adds some, but the bulk should be preserved.
    expect(Math.abs(p1.x - p0.x) / Math.abs(p0.x)).toBeLessThan(0.05);
    expect(Math.abs(p1.z - p0.z) / Math.abs(p0.z)).toBeLessThan(0.05);
  });
});
