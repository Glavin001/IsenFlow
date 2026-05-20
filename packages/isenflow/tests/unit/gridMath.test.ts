import { describe, it, expect } from 'vitest';
import {
  worldToGrid, gridToWorldCenter, clipAabbToGrid, indexOf, cellArea, isInBounds,
} from '../../src/utils/gridMath.js';

const grid = { width: 16, height: 8, dx: 2, origin: [-16, -8] as const };

describe('gridMath', () => {
  it('worldToGrid maps minimum corner to (0, 0)', () => {
    expect(worldToGrid(grid, -16, -8)).toEqual([0, 0]);
  });
  it('worldToGrid maps a center to its cell', () => {
    expect(worldToGrid(grid, -15, -7)).toEqual([0, 0]);
    expect(worldToGrid(grid, -13.9, -5.9)).toEqual([1, 1]);
  });
  it('gridToWorldCenter is the inverse of worldToGrid at cell centers', () => {
    const [x, z] = gridToWorldCenter(grid, 3, 2);
    expect([x, z]).toEqual([-16 + 3.5 * 2, -8 + 2.5 * 2]);
  });
  it('clipAabbToGrid clamps to bounds', () => {
    const r = clipAabbToGrid(grid, { minX: -1000, minZ: -1000, maxX: 1000, maxZ: 1000 });
    expect(r).toEqual({ i0: 0, j0: 0, i1: 15, j1: 7 });
  });
  it('clipAabbToGrid keeps interior boxes intact', () => {
    const r = clipAabbToGrid(grid, { minX: -10, minZ: -4, maxX: -6, maxZ: 0 });
    expect(r.i0).toBe(3);
    expect(r.j0).toBe(2);
    expect(r.i1).toBe(5);
    expect(r.j1).toBe(4);
  });
  it('indexOf is row-major', () => {
    expect(indexOf(grid, 0, 0)).toBe(0);
    expect(indexOf(grid, 15, 7)).toBe(7 * 16 + 15);
  });
  it('cellArea = dx²', () => {
    expect(cellArea(grid)).toBe(4);
  });
  it('isInBounds rejects negatives and overflow', () => {
    expect(isInBounds(grid, -1, 0)).toBe(false);
    expect(isInBounds(grid, 0, 8)).toBe(false);
    expect(isInBounds(grid, 0, 0)).toBe(true);
    expect(isInBounds(grid, 15, 7)).toBe(true);
  });
});
