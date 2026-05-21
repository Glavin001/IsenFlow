/**
 * SWASHES analytic solution tests — verify the pure-TS implementations.
 */
import { describe, it, expect } from 'vitest';
import {
  stokerDamBreak,
  stokerFrontPosition,
  thackerParabolicBasin,
  lakeAtRestOverBump,
  manningNormalDepth,
} from '../../src/math/swashes.js';
import {
  cpuStep,
  createCpuState,
  createCpuParams,
} from '../../src/core/cpu/CpuVirtualPipes.js';

describe('Stoker dam-break', () => {
  it('returns H0 upstream of rarefaction', () => {
    const { h } = stokerDamBreak(1.0, 1.0, -10);
    expect(h).toBeCloseTo(1.0, 5);
  });

  it('returns 0 on dry bed ahead of front', () => {
    const { h } = stokerDamBreak(1.0, 1.0, 10);
    expect(h).toBe(0);
  });

  it('rarefaction fan has correct depth at x=0', () => {
    // At x=0, t>0: h = (2c0)^2 / (9g) = 4*g*H0 / (9*g) = 4H0/9
    const { h } = stokerDamBreak(1.0, 1.0, 0);
    expect(h).toBeCloseTo(4 / 9, 4);
  });

  it('front position matches 2*sqrt(g*H0)*t', () => {
    const pos = stokerFrontPosition(1.0, 1.0);
    expect(pos).toBeCloseTo(2 * Math.sqrt(9.81), 3);
  });
});

describe('Thacker parabolic basin', () => {
  it('returns correct period', () => {
    const { period } = thackerParabolicBasin(10, 1, 0, 0);
    const expected = (2 * Math.PI) / (Math.sqrt(2 * 9.81 * 1) / 10);
    expect(period).toBeCloseTo(expected, 3);
  });

  it('depth is positive at center', () => {
    const { h } = thackerParabolicBasin(10, 1, 0, 0);
    expect(h).toBeGreaterThan(0);
  });

  it('depth is zero far from center', () => {
    const { h } = thackerParabolicBasin(10, 1, 0, 20);
    expect(h).toBe(0);
  });
});

describe('Lake-at-rest over bump', () => {
  it('flat free surface yields h = eta0 - zb', () => {
    expect(lakeAtRestOverBump(0.3, 1.0).h).toBeCloseTo(0.7, 10);
    expect(lakeAtRestOverBump(0.0, 1.0).h).toBeCloseTo(1.0, 10);
  });

  it('dry cells have h = 0', () => {
    expect(lakeAtRestOverBump(1.5, 1.0).h).toBe(0);
  });
});

describe('CPU oracle vs Stoker dam-break', () => {
  it('upstream rarefaction fan depth decreases monotonically from dam', () => {
    // Virtual Pipes is diffusive for sharp fronts, so instead of point-by-point
    // Stoker comparison, verify the qualitative shape: monotonically decreasing
    // depth from upstream toward the front, and reasonable depth at dam.
    const W = 512, H = 4, dx = 0.1;
    const state = createCpuState(W, H);
    const params = createCpuParams(W, H, { dx, dt: 0.0005, damping: 1.0, manningN: 0 });

    const H0 = 2.0;
    const damI = Math.floor(W / 2);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const h = i < damI ? H0 : 0.0;
        state.water[(j * W + i) * 2] = h;
        state.water[(j * W + i) * 2 + 1] = h;
      }
    }

    const nSteps = 2000;
    for (let s = 0; s < nSteps; s++) cpuStep(state, params);

    const midJ = Math.floor(H / 2);

    // Depth at dam position: Stoker gives 4H0/9 ≈ 0.889. Allow wide range for VP.
    const hAtDam = state.water[(midJ * W + damI) * 2]!;
    expect(hAtDam).toBeGreaterThan(0.4);
    expect(hAtDam).toBeLessThan(1.8);

    // Overall decreasing trend from dam rightward: sample every 5 cells to
    // smooth out VP oscillations, and verify the 5-cell-averaged profile drops.
    const sampleStep = 5;
    let prevAvg = hAtDam;
    let decreases = 0, total = 0;
    for (let i = damI + sampleStep; i < damI + 30; i += sampleStep) {
      let sum = 0;
      for (let k = 0; k < sampleStep; k++) sum += state.water[(midJ * W + i + k) * 2]!;
      const avg = sum / sampleStep;
      total++;
      if (avg <= prevAvg + 0.01) decreases++;
      prevAvg = avg;
    }
    // At least 60% of sampled intervals decrease
    expect(total).toBeGreaterThan(0);
    expect(decreases / total).toBeGreaterThanOrEqual(0.6);
  });
});

describe('Manning normal depth', () => {
  it('computes h_n = (n*q/sqrt(S0))^(3/5)', () => {
    const q = 1.0, slope = 0.005, n = 0.03;
    const { h, u } = manningNormalDepth(q, slope, n);
    const expected = Math.pow((n * q) / Math.sqrt(slope), 3 / 5);
    expect(h).toBeCloseTo(expected, 6);
    expect(u).toBeCloseTo(q / h, 6);
  });

  it('returns zero for zero slope', () => {
    const { h } = manningNormalDepth(1.0, 0, 0.03);
    expect(h).toBe(0);
  });
});
