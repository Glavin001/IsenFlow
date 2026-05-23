/**
 * Stress-bound tests for the KP CPU oracle.
 *
 * These guard against the spike-class regressions that prompted the
 * Virtual-Pipes → Kurganov–Petrova migration:
 *
 *   - "spikeBound": rain barrage + dam break + solid wall; no cell may
 *     reach height > 3× initial-max within 5 s of sim time.
 *   - "wellBalancedSteepStep": lake at rest over a 1-cell vertical bed
 *     step preserves the flat surface for 2000 steps.
 *   - "meteoriteImpactBounded": single high-Froude impact does NOT
 *     produce h > 3× waterDepth anywhere.
 *   - "solidWallMassConservation": flux around a Solid column conserves
 *     volume to 0.1%.
 *   - "noNaN": no NaNs / Infs anywhere in any state field after long runs.
 */
import { describe, it, expect } from 'vitest';
import {
  createKpState,
  createKpParams,
  createKpScratch,
  kpStep,
  BT_CLOSED,
  BT_SOLID,
  BT_INFLOW,
  type KpState,
} from '../../src/core/cpu/CpuKurganov.js';

function fillBoundaryAll(p: { boundary: Uint32Array }, type: number): void {
  p.boundary.fill(type);
}
function totalVolume(state: KpState, dx: number): number {
  let s = 0;
  for (let i = 0; i < state.h.length; i++) s += Math.max(0, state.h[i]!) * dx * dx;
  return s;
}
function maxH(state: KpState): number {
  let m = 0;
  for (let i = 0; i < state.h.length; i++) if (state.h[i]! > m) m = state.h[i]!;
  return m;
}
function hasBadNumbers(state: KpState): boolean {
  for (const arr of [state.h, state.hu, state.hv]) {
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i]!)) return true;
    }
  }
  return false;
}

describe('KP — spike bounds', () => {
  it('spikeBound: combined torture (rain + dam break + solid wall) — no h > 3× initial-max in 5 s', () => {
    const W = 48, H = 48;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.1, dt: 0.001, gravity: 9.81, manningN: 0.03 });
    fillBoundaryAll(params, BT_CLOSED);

    // West-edge inflow at 0.8 m
    for (let j = 0; j < H; j++) {
      params.boundary[j * W] = BT_INFLOW;
      params.boundaryTargetH[j * W] = 0.8;
    }

    // Solid wall column at i = W/2, middle 2/3 of the height
    const wallI = Math.floor(W / 2);
    for (let j = Math.floor(H / 6); j < Math.floor(5 * H / 6); j++) {
      params.boundary[j * W + wallI] = BT_SOLID;
    }

    // Initial dam: water on left, dry on right
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        state.h[j * W + i] = i < wallI ? 1.0 : 0.01;
      }
    }

    const scratch = createKpScratch(W, H);
    const totalSteps = Math.round(5.0 / params.dt);  // 5 s
    const initialMax = maxH(state);
    let observedMax = initialMax;
    let rainCounter = 0;

    for (let s = 0; s < totalSteps; s++) {
      // "Rain": every 20 steps, drop +0.05 m on 8 random cells
      if (rainCounter % 20 === 0) {
        for (let r = 0; r < 8; r++) {
          const ri = Math.floor(((s * 7919 + r * 137) % 10007) / 10007 * W);
          const rj = Math.floor(((s * 6151 + r * 113) % 10007) / 10007 * H);
          const c = rj * W + ri;
          if (params.boundary[c] !== BT_SOLID) {
            state.h[c] += 0.05;
          }
        }
      }
      rainCounter++;

      kpStep(state, params, scratch);
      const cur = maxH(state);
      if (cur > observedMax) observedMax = cur;
    }

    expect(hasBadNumbers(state), 'no NaN/Inf').toBe(false);
    // After 5s of relentless rain + inflow + dam pushing against a wall,
    // no cell should have spiked above ~3 m (well below the 3×initial=3m
    // VP would routinely violate with shafts of 10m+).
    expect(observedMax).toBeLessThan(Math.max(3 * initialMax, 3.0));
  });

  it('lake-at-rest over vertical wall: 2000 steps, |u| stays microscopic', () => {
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.1, dt: 0.001, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);

    // Wall: bed steps up by 1.5 m in the middle, water surface flat at 3 m
    const eta0 = 3.0;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const bed = (i >= W / 2) ? 1.5 : 0;
        state.bed[j * W + i] = bed;
        state.h[j * W + i] = Math.max(0, eta0 - bed);
      }
    }

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 2000; s++) kpStep(state, params, scratch);

    let maxAbsHu = 0;
    let maxAbsHv = 0;
    for (let i = 0; i < state.h.length; i++) {
      if (Math.abs(state.hu[i]!) > maxAbsHu) maxAbsHu = Math.abs(state.hu[i]!);
      if (Math.abs(state.hv[i]!) > maxAbsHv) maxAbsHv = Math.abs(state.hv[i]!);
    }
    expect(maxAbsHu).toBeLessThan(1e-3);
    expect(maxAbsHv).toBeLessThan(1e-3);

    // Verify the surface remained flat
    for (let c = 0; c < W * H; c++) {
      const eta = state.h[c]! + state.bed[c]!;
      expect(Math.abs(eta - eta0)).toBeLessThan(0.05);
    }
  });

  it('meteorite-class impulse: ring + crater never exceeds 3× water depth', () => {
    const W = 64, H = 64;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.1, dt: 0.001, gravity: 9.81, manningN: 0.03 });
    fillBoundaryAll(params, BT_CLOSED);

    const waterDepth = 1.5;
    for (let i = 0; i < state.h.length; i++) state.h[i] = waterDepth;

    // Inject a meteorite-class crater + ring + outward momentum (mimicking
    // ImpactDisplacement's output)
    const ci = W / 2, cj = H / 2;
    const craterRad = 4;
    const ringRad = 10;
    for (let dj = -ringRad; dj <= ringRad; dj++) {
      for (let di = -ringRad; di <= ringRad; di++) {
        const r = Math.hypot(di, dj);
        const c = (cj + dj) * W + (ci + di);
        if (r < craterRad) {
          const t = r / craterRad;
          state.h[c] = Math.max(0.01, waterDepth - 0.5 * waterDepth * 0.5 * (1 + Math.cos(Math.PI * t)));
        } else if (r < ringRad) {
          const t = (r - craterRad) / (ringRad - craterRad);
          state.h[c] = waterDepth + 0.5 * waterDepth * 0.5 * (1 + Math.cos(Math.PI * t));
          // Outward radial momentum
          const speed = 2.0;
          const dist = Math.max(1e-6, r);
          state.hu[c] = state.h[c]! * speed * (di / dist);
          state.hv[c] = state.h[c]! * speed * (dj / dist);
        }
      }
    }

    const scratch = createKpScratch(W, H);
    let observedMax = maxH(state);
    for (let s = 0; s < 500; s++) {
      kpStep(state, params, scratch);
      const cur = maxH(state);
      if (cur > observedMax) observedMax = cur;
    }
    expect(hasBadNumbers(state)).toBe(false);
    expect(observedMax).toBeLessThan(3 * waterDepth);
  });

  it('Solid column conserves total mass within 0.1 % over 1000 steps', () => {
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.1, dt: 0.001, gravity: 9.81, manningN: 0 });
    fillBoundaryAll(params, BT_CLOSED);

    // Single solid column at center
    const ci = W / 2, cj = H / 2;
    params.boundary[cj * W + ci] = BT_SOLID;

    // Asymmetric water — gradient should drive flow around the solid
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const c = j * W + i;
        if (params.boundary[c] === BT_SOLID) continue;
        state.h[c] = 1.0 + 0.3 * Math.exp(-((i - W / 4) ** 2 + (j - H / 4) ** 2) / 16);
      }
    }

    const v0 = totalVolume(state, params.dx);
    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 1000; s++) kpStep(state, params, scratch);
    const v1 = totalVolume(state, params.dx);

    expect(hasBadNumbers(state)).toBe(false);
    expect(Math.abs(v1 - v0) / v0).toBeLessThan(0.001);
    // Solid still empty
    expect(state.h[cj * W + ci]).toBe(0);
  });

  it('no NaN/Inf after 10000 steps of mixed inflow + dam + closed walls', () => {
    const W = 32, H = 32;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx: 0.1, dt: 0.001, gravity: 9.81, manningN: 0.03 });
    fillBoundaryAll(params, BT_CLOSED);
    for (let j = 0; j < H; j++) {
      params.boundary[j * W] = BT_INFLOW;
      params.boundaryTargetH[j * W] = 1.5;
    }
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        state.h[j * W + i] = i < W / 3 ? 2.0 : 0.05;
      }
    }
    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 10000; s++) kpStep(state, params, scratch);
    expect(hasBadNumbers(state)).toBe(false);
  });
});
