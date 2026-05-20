import { describe, it, expect } from 'vitest';
import {
  BoundaryType, isReflective, isAbsorbing, isSourceLike,
} from '../../src/boundaries/BoundaryConditions.js';

describe('BoundaryConditions', () => {
  it('enum values are stable integers', () => {
    expect(BoundaryType.Interior).toBe(0);
    expect(BoundaryType.Closed).toBe(1);
    expect(BoundaryType.Open).toBe(2);
    expect(BoundaryType.Sponge).toBe(3);
    expect(BoundaryType.Inflow).toBe(4);
    expect(BoundaryType.Sea).toBe(5);
  });

  it('isReflective is true only for Closed', () => {
    expect(isReflective(BoundaryType.Closed)).toBe(true);
    expect(isReflective(BoundaryType.Open)).toBe(false);
    expect(isReflective(BoundaryType.Interior)).toBe(false);
  });

  it('isAbsorbing covers Open and Sponge', () => {
    expect(isAbsorbing(BoundaryType.Open)).toBe(true);
    expect(isAbsorbing(BoundaryType.Sponge)).toBe(true);
    expect(isAbsorbing(BoundaryType.Closed)).toBe(false);
  });

  it('isSourceLike covers Inflow and Sea', () => {
    expect(isSourceLike(BoundaryType.Inflow)).toBe(true);
    expect(isSourceLike(BoundaryType.Sea)).toBe(true);
    expect(isSourceLike(BoundaryType.Interior)).toBe(false);
  });
});
