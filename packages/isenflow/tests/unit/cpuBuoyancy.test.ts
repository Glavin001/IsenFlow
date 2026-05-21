import { describe, it, expect } from 'vitest';
import {
  computeCpuBuoyancy,
  computeVerticalDamping,
  estimateFootprintCells,
} from '../../src/coupling/CpuBuoyancy.js';
import { WATER_DENSITY, GRAVITY } from '../../src/coupling/HydrostaticMath.js';

describe('CpuBuoyancy', () => {
  const dx = 0.0417; // ~384 cells in 16m

  describe('computeCpuBuoyancy', () => {
    it('returns zero when body is above water', () => {
      const result = computeCpuBuoyancy({
        comY: 5.0,
        halfY: 0.1,
        linvelY: 0,
        waterLevel: 1.5,
        bedLevel: 0,
        footprintCells: 100,
        dx,
      });
      expect(result.fy).toBe(0);
      expect(result.submergedFraction).toBe(0);
    });

    it('returns full buoyancy when fully submerged', () => {
      const halfY = 0.1;
      const result = computeCpuBuoyancy({
        comY: 0.5,   // body from 0.4 to 0.6
        halfY,
        linvelY: 0,
        waterLevel: 2.0, // water well above body top
        bedLevel: 0,
        footprintCells: 1,
        dx,
      });
      // submerged_h = min(2.0, 0.6) - max(0, 0.4) = 0.6 - 0.4 = 0.2 (full body height)
      expect(result.submergedFraction).toBeCloseTo(1.0);
      const expectedForce = WATER_DENSITY * GRAVITY * dx * dx * 0.2 * 1;
      expect(result.fy).toBeCloseTo(expectedForce);
    });

    it('computes partial submersion correctly (Archimedes equilibrium)', () => {
      // A box of density 400 in water of density 1000 should float with
      // 40% of its volume submerged at equilibrium.
      const halfY = 0.1; // total height 0.2m
      // At equilibrium, submergedFraction = 400/1000 = 0.4
      // submerged_h = 0.4 * 0.2 = 0.08m
      // body bottom at equilibrium: comY - halfY
      // waterLevel = bedLevel + waterDepth
      // For equilibrium: comY = waterLevel - (1 - 0.4)*halfY*2 ... let's just set it.
      // submerged_h = min(waterLevel, bodyTop) - max(bedLevel, bodyBot)
      // = min(1.0, comY+0.1) - max(0, comY-0.1)
      // If comY = 0.92: bodyTop=1.02, bodyBot=0.82
      // submerged_h = min(1.0, 1.02) - max(0, 0.82) = 1.0 - 0.82 = 0.18
      // fraction = 0.18/0.2 = 0.9 -- too much. Need comY higher.
      // At equilibrium for 40% submerged: submerged_h = 0.08
      // waterLevel=1.0; bodyTop = comY + 0.1; bodyBot = comY - 0.1
      // 0.08 = min(1.0, comY+0.1) - max(0, comY-0.1)
      // If comY > 0.9 (bodyTop > 1.0): 0.08 = 1.0 - (comY-0.1) => comY = 1.02
      // bodyBot = 0.92, bodyTop = 1.12 -> submerged_h = min(1.0, 1.12) - max(0, 0.92) = 0.08 ✓
      const result = computeCpuBuoyancy({
        comY: 1.02,
        halfY: 0.1,
        linvelY: 0,
        waterLevel: 1.0,
        bedLevel: 0,
        footprintCells: 1,
        dx,
      });
      expect(result.submergedFraction).toBeCloseTo(0.4, 1);
    });

    it('handles body partially below bed', () => {
      // Body COM below bed level — should only count portion above bed
      const result = computeCpuBuoyancy({
        comY: -0.05,
        halfY: 0.1,
        linvelY: 0,
        waterLevel: 1.0,
        bedLevel: 0,
        footprintCells: 1,
        dx,
      });
      // bodyBot = -0.15, but max(bedLevel, bodyBot) = 0
      // bodyTop = 0.05, submerged_h = min(1.0, 0.05) - 0 = 0.05
      expect(result.fy).toBeGreaterThan(0);
    });
  });

  describe('computeVerticalDamping', () => {
    it('returns zero when not submerged', () => {
      expect(computeVerticalDamping(5.0, 0, dx, 100)).toBe(0);
    });

    it('opposes downward motion', () => {
      const force = computeVerticalDamping(-2.0, 0.5, dx, 100);
      expect(force).toBeGreaterThan(0); // positive = upward, opposing downward
    });

    it('opposes upward motion', () => {
      const force = computeVerticalDamping(2.0, 0.5, dx, 100);
      expect(force).toBeLessThan(0); // negative = downward, opposing upward
    });

    it('scales with submersion fraction', () => {
      const half = computeVerticalDamping(-1.0, 0.5, dx, 100);
      const full = computeVerticalDamping(-1.0, 1.0, dx, 100);
      expect(full).toBeCloseTo(half * 2);
    });
  });

  describe('estimateFootprintCells', () => {
    it('computes correct cell count for small body', () => {
      // 0.4m x 0.4m body on dx=0.0417 grid
      // ceil(0.4/0.0417) = ceil(9.59) = 10; 10*10 = 100
      const cells = estimateFootprintCells(0.2, 0.2, dx);
      expect(cells).toBe(100);
    });

    it('returns at least 1', () => {
      expect(estimateFootprintCells(0.001, 0.001, 1.0)).toBe(1);
    });
  });
});
