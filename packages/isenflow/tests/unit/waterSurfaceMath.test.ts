/**
 * Unit tests for the pure water-surface math.
 *
 * These tests are the "spec" for the spike-free rendering pipeline:
 *   - no Y cliffs between wet vertices on a smooth depth field,
 *   - sub-cell shorelines fade via per-vertex alpha (wetness ∈ [0,1]),
 *   - 3x3 blur kills a checkerboard while preserving a Gaussian bump,
 *   - analytic normals stay sane across a wet/dry boundary that would
 *     otherwise leak terrain slopes into the water lighting.
 */
import { describe, it, expect } from 'vitest';
import {
  computeWaterVertices,
  computeWaterNormals,
  smoothScalarField,
  extractWaterDepth,
  extractBedTotal,
  lerpFieldInPlace,
} from '../../src/render/waterSurfaceMath.js';

function checkerboard(w: number, h: number, lo: number, hi: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      out[j * w + i] = ((i + j) & 1) === 0 ? lo : hi;
    }
  }
  return out;
}

function gaussianBump(w: number, h: number, peak: number, sigmaCells: number): Float32Array {
  const out = new Float32Array(w * h);
  const cx = (w - 1) * 0.5;
  const cy = (h - 1) * 0.5;
  const s2 = 2 * sigmaCells * sigmaCells;
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const dx = i - cx, dy = j - cy;
      out[j * w + i] = peak * Math.exp(-(dx * dx + dy * dy) / s2);
    }
  }
  return out;
}

function variance(a: Float32Array): number {
  let mean = 0;
  for (const v of a) mean += v;
  mean /= a.length;
  let s = 0;
  for (const v of a) {
    const d = v - mean;
    s += d * d;
  }
  return s / a.length;
}

describe('extractWaterDepth / extractBedTotal', () => {
  it('pulls every second value, clamps depth to >= 0', () => {
    // [h, h_prev] interleaved.
    const interleaved = new Float32Array([1.5, 9, -0.2, 9, 0, 9, 4.0, 9]);
    const out = new Float32Array(4);
    extractWaterDepth(interleaved, out);
    expect(Array.from(out)).toEqual([1.5, 0, 0, 4]);
  });
  it('bedTotal pulls .y (total bed) and preserves sign', () => {
    // [terrain, total] interleaved; total can be negative (valley).
    const interleaved = new Float32Array([0.4, 0.4, 0.4, -0.3, 0.0, 6.0]);
    const out = new Float32Array(3);
    extractBedTotal(interleaved, out);
    expect(out[0]).toBeCloseTo(0.4, 5);
    expect(out[1]).toBeCloseTo(-0.3, 5);
    expect(out[2]).toBeCloseTo(6.0, 5);
  });
});

describe('smoothScalarField', () => {
  it('iter=0 returns a copy untouched', () => {
    const f = new Float32Array([1, 2, 3, 4]);
    const s = smoothScalarField(f, 2, 2, 0);
    expect(s).not.toBe(f);
    expect(Array.from(s)).toEqual([1, 2, 3, 4]);
  });

  it('kills checkerboard variance', () => {
    const w = 16, h = 16;
    const cb = checkerboard(w, h, 0, 1);
    const var0 = variance(cb);
    const s = smoothScalarField(cb, w, h, 1);
    const var1 = variance(s);
    // 3x3 box blur over a 0/1 checkerboard collapses variance ~10×.
    expect(var1).toBeLessThan(var0 * 0.2);
  });

  it('preserves a smooth Gaussian bump (peak attenuation < 10%)', () => {
    const w = 32, h = 32;
    const bump = gaussianBump(w, h, 1, 6);
    const s = smoothScalarField(bump, w, h, 1);
    const peakOrig = Math.max(...bump);
    const peakSmooth = Math.max(...s);
    expect(peakSmooth).toBeGreaterThan(peakOrig * 0.9);
  });

  it('is approximately mass-preserving when the bump is well inside the grid', () => {
    // With clamped edges, mass leaks out at the boundary as the blur
    // averages with replicated edge values. Use a 32x32 grid where the
    // bump is far from the edges so leakage is negligible.
    const w = 32, h = 32;
    const bump = gaussianBump(w, h, 2, 2);
    let s0 = 0; for (const v of bump) s0 += v;
    const s = smoothScalarField(bump, w, h, 3);
    let s1 = 0; for (const v of s) s1 += v;
    expect(Math.abs(s1 - s0) / s0).toBeLessThan(0.01);
  });

  it('throws on length mismatch', () => {
    expect(() => smoothScalarField(new Float32Array(3), 2, 2, 1)).toThrow();
  });
});

describe('computeWaterVertices', () => {
  it('flat lake: every wet vertex sits at the same Y, no cliffs', () => {
    const w = 8, h = 8;
    const depth = new Float32Array(w * h).fill(1.0);
    const bed = new Float32Array(w * h).fill(0.4);
    const { positionsY, wetness, maxWetSpike } = computeWaterVertices(depth, bed, w, h);
    // Interior wet verts must all be 1.4 (bed + h).
    for (let j = 1; j < h; j++) {
      for (let i = 1; i < w; i++) {
        const vi = j * (w + 1) + i;
        expect(positionsY[vi]).toBeCloseTo(1.4, 5);
        expect(wetness[vi]).toBe(1);
      }
    }
    expect(maxWetSpike).toBeLessThan(1e-6);
  });

  it('dry vertex hugs bed and is fully transparent', () => {
    const w = 4, h = 4;
    const depth = new Float32Array(w * h); // all dry
    const bed = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) bed[i] = 0.5 + 0.1 * i; // varying terrain
    const { positionsY, wetness } = computeWaterVertices(depth, bed, w, h);
    for (const a of wetness) expect(a).toBe(0);
    // No vertex parks at the legacy -1000 sentinel.
    for (const y of positionsY) {
      expect(y).toBeGreaterThan(0);
      expect(y).toBeLessThan(10);
    }
  });

  it('shoreline: wet-side eta does not get dragged down by a dry-bed neighbour', () => {
    // 2×1 grid: cell 0 has water 2 m on bed 0; cell 1 is dry with bed -5.
    // The corner between them is wet because cell 0 is wet — Y must equal
    // bed_wet + h = 0 + 2 = 2, NOT the average with the dry bed = -1.5.
    const w = 2, h = 1;
    const depth = new Float32Array([2, 0]);
    const bed = new Float32Array([0, -5]);
    const { positionsY, wetness } = computeWaterVertices(depth, bed, w, h);
    // Corner vertex (i=1, j=0) borders cell 0 (wet) and cell 1 (dry).
    const corner = positionsY[0 * (w + 1) + 1]!;
    expect(corner).toBeCloseTo(2, 5);
    expect(wetness[0 * (w + 1) + 1]).toBeGreaterThan(0);
    expect(wetness[0 * (w + 1) + 1]).toBeLessThan(1); // partial fill
  });

  it('legacy regression: sharp wet/dry boundary has no wet-wet cliff > local h', () => {
    // Half the grid has 1 m of water, the other half is dry. The new
    // sampler must never produce a wet-vertex jump greater than the local
    // wave amplitude. (The legacy code's -1000 sentinel would have produced
    // a ~1000 m wet→dry jump that polluted normals.)
    const w = 16, h = 16;
    const depth = new Float32Array(w * h);
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        depth[j * w + i] = i < w / 2 ? 1 : 0;
      }
    }
    const bed = new Float32Array(w * h).fill(0);
    const { maxWetSpike } = computeWaterVertices(depth, bed, w, h);
    // The biggest wet-wet jump is between two corner vertices that have
    // different "fillFrac" weights but identical wet-eta=1.0, so the only
    // remaining wet-wet delta is identically zero.
    expect(maxWetSpike).toBeLessThan(0.5);
  });

  it('wetness ramps smoothly from 0 to 1 across a thin-water shoreline', () => {
    // Linear depth ramp 0..0.05 m across 10 cells. Wetness should rise
    // monotonically from 0 to 1.
    const w = 10, h = 1;
    const depth = new Float32Array(w);
    for (let i = 0; i < w; i++) depth[i] = (i / (w - 1)) * 0.05;
    const bed = new Float32Array(w);
    const { wetness } = computeWaterVertices(depth, bed, w, h, {
      wetThreshold: 0.003,
      fadeBandFactor: 6,
    });
    let last = -1;
    for (let i = 0; i < w + 1; i++) {
      const a = wetness[0 * (w + 1) + i]!;
      expect(a).toBeGreaterThanOrEqual(last - 1e-6); // non-decreasing
      last = a;
    }
    expect(wetness[0]).toBe(0);
    expect(wetness[w]).toBeCloseTo(1, 3);
  });

  it('reuses scratch buffers when sizes match', () => {
    const w = 4, h = 4;
    const depth = new Float32Array(w * h);
    const bed = new Float32Array(w * h);
    const scratch = {
      positionsY: new Float32Array((w + 1) * (h + 1)),
      wetness: new Float32Array((w + 1) * (h + 1)),
    };
    const r = computeWaterVertices(depth, bed, w, h, {}, scratch);
    expect(r.positionsY).toBe(scratch.positionsY);
    expect(r.wetness).toBe(scratch.wetness);
  });

  it('throws on length mismatch', () => {
    expect(() =>
      computeWaterVertices(new Float32Array(3), new Float32Array(4), 2, 2),
    ).toThrow();
  });
});

describe('computeWaterNormals', () => {
  it('flat lake → all up-normals', () => {
    const w = 4, h = 4;
    const vw = w + 1, vh = h + 1;
    const Y = new Float32Array(vw * vh).fill(1);
    const W = new Float32Array(vw * vh).fill(1);
    const N = computeWaterNormals(Y, W, vw, vh, 0.1);
    for (let i = 0; i < vw * vh; i++) {
      expect(N[i * 3]).toBeCloseTo(0, 5);
      expect(N[i * 3 + 1]).toBeCloseTo(1, 5);
      expect(N[i * 3 + 2]).toBeCloseTo(0, 5);
    }
  });

  it('tilted sheet → consistent slanted normal', () => {
    const w = 4, h = 4;
    const vw = w + 1, vh = h + 1;
    const Y = new Float32Array(vw * vh);
    const dx = 0.1;
    for (let j = 0; j < vh; j++) {
      for (let i = 0; i < vw; i++) {
        Y[j * vw + i] = i * dx; // slope of 1 in x.
      }
    }
    const W = new Float32Array(vw * vh).fill(1);
    const N = computeWaterNormals(Y, W, vw, vh, dx);
    // Interior vertex: gradient = (1, 0), normal = normalize(-1, 1, 0).
    const vi = 2 * vw + 2;
    const inv = 1 / Math.SQRT2;
    expect(N[vi * 3]).toBeCloseTo(-inv, 4);
    expect(N[vi * 3 + 1]).toBeCloseTo(inv, 4);
    expect(N[vi * 3 + 2]).toBeCloseTo(0, 5);
  });

  it('dry neighbours never leak terrain slope into wet normals', () => {
    // Imagine a wet vertex bordered by a wet neighbour at the same height
    // and a "dry" neighbour parked on a 1 m terrain hump. With the wet
    // mask in place, the wet vertex's normal must still be up — the dry
    // terrain hump is invisible to the gradient.
    const vw = 3, vh = 1;
    const Y = new Float32Array([0, 0, 1]); // 3rd vertex sits on hump
    const W = new Float32Array([1, 1, 0]); // 3rd is dry
    const N = computeWaterNormals(Y, W, vw, vh, 0.1);
    // Vertex 1 (middle, wet): the right neighbour is dry → centre Y reused.
    expect(N[1 * 3]).toBeCloseTo(0, 5);
    expect(N[1 * 3 + 1]).toBeCloseTo(1, 5);
  });
});

describe('lerpFieldInPlace', () => {
  it('alpha=0 is a no-op', () => {
    const prev = new Float32Array([1, 2, 3]);
    lerpFieldInPlace(prev, new Float32Array([9, 9, 9]), 0);
    expect(Array.from(prev)).toEqual([1, 2, 3]);
  });
  it('alpha=1 snaps to target', () => {
    const prev = new Float32Array([1, 2, 3]);
    lerpFieldInPlace(prev, new Float32Array([9, 8, 7]), 1);
    expect(Array.from(prev)).toEqual([9, 8, 7]);
  });
  it('alpha=0.5 halves the gap', () => {
    const prev = new Float32Array([0, 0, 0]);
    lerpFieldInPlace(prev, new Float32Array([4, 4, 4]), 0.5);
    expect(prev[0]).toBeCloseTo(2);
  });
  it('alpha is clamped to [0,1]', () => {
    const prev = new Float32Array([0]);
    lerpFieldInPlace(prev, new Float32Array([1]), 2);
    expect(prev[0]).toBe(1);
    lerpFieldInPlace(prev, new Float32Array([0]), -1);
    expect(prev[0]).toBe(1); // unchanged from clamp to 0
  });
});
