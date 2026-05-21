/**
 * Tests for boundary-driven flow using the CPU oracle.
 *
 * These tests verify that Sea boundaries at different levels create steady
 * flow in the correct direction — a key regression test for the riverboat demo.
 */
import { describe, it, expect } from 'vitest';
import {
  cpuStep,
  createCpuState,
  createCpuParams,
} from '../../src/core/cpu/CpuVirtualPipes.js';

// BoundaryType values (matching BoundaryConditions.ts)
const BT_SEA = 5;

describe('Boundary-driven river flow (CPU oracle)', () => {
  it('Sea-Sea head difference drives eastward flow', () => {
    // 1D channel: 32 cells wide, 3 cells tall (banks at top/bottom)
    const W = 32, H = 3;
    const dx = 0.5; // 16m world
    const params = createCpuParams(W, H, { dx, damping: 0.9, manningN: 0.03 });
    const state = createCpuState(W, H);

    const hWest = 1.0; // higher
    const hEast = 0.8; // lower

    // Fill channel (row 1) with water at average depth
    const hInit = 0.9;
    for (let i = 0; i < W; i++) {
      const ci = 1 * W + i; // row 1 (channel)
      state.water[ci * 2] = hInit;
    }

    // Banks: row 0 and row 2 have high bed
    for (let i = 0; i < W; i++) {
      state.bed[(0 * W + i) * 2 + 1] = 10; // bed_total
      state.bed[(2 * W + i) * 2 + 1] = 10;
    }

    // Sea boundary at west (col 0) and east (col W-1)
    params.boundary[1 * W + 0] = BT_SEA;
    params.boundaryTargetH[1 * W + 0] = hWest;
    params.boundary[1 * W + (W - 1)] = BT_SEA;
    params.boundaryTargetH[1 * W + (W - 1)] = hEast;

    // Run many steps to establish flow
    for (let t = 0; t < 5000; t++) {
      cpuStep(state, params);
    }

    // Check velocity at mid-channel — should be positive (eastward)
    const midI = Math.floor(W / 2);
    const midIdx = 1 * W + midI;
    const u = state.velocity[midIdx * 2]!;
    expect(u, 'mid-channel u velocity should be positive (eastward)').toBeGreaterThan(0);

    // Average velocity across channel should be positive
    let sumU = 0;
    for (let i = 2; i < W - 2; i++) {
      sumU += state.velocity[(1 * W + i) * 2]!;
    }
    const meanU = sumU / (W - 4);
    expect(meanU, 'mean channel velocity should be eastward').toBeGreaterThan(0);
  });

  it('reversed Sea-Sea head difference drives westward flow', () => {
    const W = 32, H = 3;
    const dx = 0.5;
    const params = createCpuParams(W, H, { dx, damping: 0.9, manningN: 0.03 });
    const state = createCpuState(W, H);

    // Now east is higher, west is lower
    const hWest = 0.8;
    const hEast = 1.0;
    const hInit = 0.9;

    for (let i = 0; i < W; i++) {
      state.water[(1 * W + i) * 2] = hInit;
      state.bed[(0 * W + i) * 2 + 1] = 10;
      state.bed[(2 * W + i) * 2 + 1] = 10;
    }

    params.boundary[1 * W + 0] = BT_SEA;
    params.boundaryTargetH[1 * W + 0] = hWest;
    params.boundary[1 * W + (W - 1)] = BT_SEA;
    params.boundaryTargetH[1 * W + (W - 1)] = hEast;

    for (let t = 0; t < 5000; t++) {
      cpuStep(state, params);
    }

    const midI = Math.floor(W / 2);
    const u = state.velocity[(1 * W + midI) * 2]!;
    expect(u, 'mid-channel u velocity should be negative (westward)').toBeLessThan(0);
  });

  it('equal Sea levels produce no net flow', () => {
    const W = 32, H = 3;
    const dx = 0.5;
    const params = createCpuParams(W, H, { dx, damping: 0.9, manningN: 0.03 });
    const state = createCpuState(W, H);

    const h = 0.8;
    for (let i = 0; i < W; i++) {
      state.water[(1 * W + i) * 2] = h;
      state.bed[(0 * W + i) * 2 + 1] = 10;
      state.bed[(2 * W + i) * 2 + 1] = 10;
    }

    params.boundary[1 * W + 0] = BT_SEA;
    params.boundaryTargetH[1 * W + 0] = h;
    params.boundary[1 * W + (W - 1)] = BT_SEA;
    params.boundaryTargetH[1 * W + (W - 1)] = h;

    for (let t = 0; t < 2000; t++) {
      cpuStep(state, params);
    }

    // Mean velocity should be essentially zero
    let sumU = 0;
    for (let i = 2; i < W - 2; i++) {
      sumU += Math.abs(state.velocity[(1 * W + i) * 2]!);
    }
    const meanAbsU = sumU / (W - 4);
    expect(meanAbsU, 'no head difference → near-zero flow').toBeLessThan(0.01);
  });

  it('Sea boundary pins water depth to target each step', () => {
    const W = 8, H = 1;
    const params = createCpuParams(W, H, { dx: 1 });
    const state = createCpuState(W, H);

    // Pin cell 0 to h=2.0
    params.boundary[0] = BT_SEA;
    params.boundaryTargetH[0] = 2.0;

    // Fill with h=1.0
    for (let i = 0; i < W; i++) {
      state.water[i * 2] = 1.0;
    }

    cpuStep(state, params);

    // Sea cell should be pinned to 2.0
    expect(state.water[0]).toBeCloseTo(2.0);
    // Interior cells should still be close to initial
    expect(state.water[2 * 2]).toBeGreaterThan(0.5);
  });
});
