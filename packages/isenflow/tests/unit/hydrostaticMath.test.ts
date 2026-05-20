import { describe, it, expect } from 'vitest';
import {
  cellBuoyancy,
  hydrostaticHorizontalForce,
  cellDragForce,
  verticalDamping,
  damBreakFrontPosition,
  cross3,
  WATER_DENSITY,
  GRAVITY,
} from '../../src/coupling/HydrostaticMath.js';

describe('HydrostaticMath', () => {
  describe('cellBuoyancy', () => {
    it('returns zero when there is no water', () => {
      expect(cellBuoyancy(0, 1, 0, 1)).toBe(0);
    });

    it('returns zero when body is above water surface', () => {
      // h=1, bodyTop=10, cellFloor=2 -> submerged depth = min(1, 10-2) = 1
      // Wait: above means bodyTop < cellFloor, no submergence.
      expect(cellBuoyancy(1, 2, 5, 1)).toBe(0);
    });

    it('matches ρ·g·V when fully submerged', () => {
      // Fully submerged volume = dx² * h
      const F = cellBuoyancy(0.5, 10, 0, 2);
      expect(F).toBeCloseTo(WATER_DENSITY * GRAVITY * 4 * 0.5, 6);
    });

    it('caps at the body top when h > body height', () => {
      // h=5, bodyTop=1, cellFloor=0 -> min(5, 1) = 1
      const F = cellBuoyancy(5, 1, 0, 1);
      expect(F).toBeCloseTo(WATER_DENSITY * GRAVITY * 1 * 1, 6);
    });
  });

  describe('hydrostaticHorizontalForce', () => {
    it('is zero when depths match', () => {
      expect(hydrostaticHorizontalForce(2, 2, 1)).toBe(0);
    });

    it('matches ½ρg(h₁²-h₂²)·dx', () => {
      // West depth 3, east depth 1, dx=2 -> 0.5*1000*9.81*(9-1)*2 = 78480
      expect(hydrostaticHorizontalForce(3, 1, 2)).toBeCloseTo(78480, 1);
    });

    it('flips sign when east > west', () => {
      const f = hydrostaticHorizontalForce(1, 3, 1);
      expect(f).toBeLessThan(0);
    });

    it('treats negative depths as zero', () => {
      expect(hydrostaticHorizontalForce(-1, -2, 1)).toBe(0);
    });
  });

  describe('cellDragForce', () => {
    it('is zero with no relative motion', () => {
      const [fx, fz] = cellDragForce(1, 5, 5, 5, 5, 1);
      expect(fx).toBe(0);
      expect(fz).toBe(0);
    });

    it('is zero with zero water depth', () => {
      const [fx, fz] = cellDragForce(0, 5, 0, 0, 0, 1);
      expect(fx).toBe(0);
      expect(fz).toBe(0);
    });

    it('points along relative velocity', () => {
      const [fx, fz] = cellDragForce(1, 5, 0, 0, 0, 1);
      expect(fx).toBeGreaterThan(0);
      expect(fz).toBe(0);
    });

    it('scales with speed²', () => {
      const [fx1] = cellDragForce(1, 1, 0, 0, 0, 1);
      const [fx2] = cellDragForce(1, 2, 0, 0, 0, 1);
      // F ∝ |v| * v = v²
      expect(fx2 / fx1).toBeCloseTo(4, 3);
    });
  });

  describe('verticalDamping', () => {
    it('opposes downward motion', () => {
      const f = verticalDamping(-2, 0.5, 1, 5000);
      expect(f).toBeGreaterThan(0);
    });
    it('is zero out of water', () => {
      expect(verticalDamping(-2, 0, 1)).toBe(0);
    });
  });

  describe('damBreakFrontPosition', () => {
    it('matches Stoker formula at t=1, H=1 -> 2√g', () => {
      expect(damBreakFrontPosition(1, 1)).toBeCloseTo(2 * Math.sqrt(GRAVITY), 6);
    });
    it('grows linearly in t', () => {
      expect(damBreakFrontPosition(1, 2)).toBeCloseTo(damBreakFrontPosition(1, 1) * 2, 6);
    });
  });

  describe('cross3', () => {
    it('matches the textbook cross product', () => {
      expect(cross3(1, 0, 0, 0, 1, 0)).toEqual([0, 0, 1]);
      expect(cross3(0, 1, 0, 0, 0, 1)).toEqual([1, 0, 0]);
    });
  });
});
