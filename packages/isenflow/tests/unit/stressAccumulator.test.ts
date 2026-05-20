import { describe, it, expect } from 'vitest';
import { createStress, tickStress } from '../../src/destruction/StressAccumulator.js';

describe('StressAccumulator', () => {
  it('does not fracture under a transient single-frame spike', () => {
    const s = createStress(1000, 0.5);
    expect(tickStress(s, 500, 1 / 60)).toBe(false);
    expect(s.fractured).toBe(false);
  });

  it('fractures under sustained heavy load', () => {
    // Steady-state accum under decay d, force F, dt: F*dt / (1 - d).
    // F=500, dt=1/60, d=0.95 -> ~166.7 > budget 50.
    const s = createStress(50, 0.95);
    let fracturedAt = -1;
    for (let i = 0; i < 600; i++) {
      if (tickStress(s, 500, 1 / 60)) { fracturedAt = i; break; }
    }
    expect(fracturedAt).toBeGreaterThan(0);
    expect(s.fractured).toBe(true);
  });

  it('only emits the fracture event on the rising edge', () => {
    const s = createStress(1, 1.0);
    expect(tickStress(s, 1000, 1)).toBe(true);
    expect(tickStress(s, 1000, 1)).toBe(false); // already fractured
  });

  it('decays toward zero under no load', () => {
    const s = createStress(100, 0.5);
    tickStress(s, 50, 1 / 60); // raise accum
    const before = s.accum;
    for (let i = 0; i < 100; i++) tickStress(s, 0, 1 / 60);
    expect(s.accum).toBeLessThan(before * 1e-6);
  });
});
