import { describe, it, expect } from 'vitest';
import { encodeFixed, decodeFixed, FIXED_POINT_SCALE } from '../../src/utils/fixedPoint.js';

describe('fixedPoint', () => {
  it('round-trips integer-resolution values exactly', () => {
    for (const v of [0, 1, -1, 100, -100, 1234, -1234]) {
      expect(decodeFixed(encodeFixed(v))).toBe(v);
    }
  });

  it('preserves sub-newton precision', () => {
    const v = 1.2345;
    expect(decodeFixed(encodeFixed(v))).toBeCloseTo(v, 4);
  });

  it('uses the documented scale', () => {
    expect(FIXED_POINT_SCALE).toBe(1e4);
    expect(encodeFixed(1)).toBe(1e4);
  });

  it('handles negative numbers symmetrically', () => {
    const v = -42.5;
    expect(decodeFixed(encodeFixed(v))).toBeCloseTo(v, 4);
  });
});
