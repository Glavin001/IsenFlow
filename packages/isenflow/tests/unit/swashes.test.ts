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
