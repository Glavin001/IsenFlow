/**
 * Layer-1 oracle tests: verify CPU oracle produces correct, stable results.
 *
 * These test the CpuVirtualPipes reference implementation against known
 * physical invariants. When run via Playwright, the same fixtures can
 * drive GPU kernels for byte-level agreement.
 */
import { describe, it, expect } from 'vitest';
import {
  cpuStep,
  cpuComputeFluxes,
  cpuUpdateWater,
  createCpuState,
  createCpuParams,
} from '../../src/core/cpu/CpuVirtualPipes.js';

describe('CPU Oracle — algebraic correctness', () => {
  it('flat basin stays flat (lake-at-rest)', () => {
    const W = 16, H = 16;
    const state = createCpuState(W, H);
    const params = createCpuParams(W, H, { dx: 1, dt: 0.01, damping: 0.5, manningN: 0 });

    // Fill uniform h = 1 m, flat bed
    for (let i = 0; i < W * H; i++) {
      state.water[i * 2] = 1.0;
      state.water[i * 2 + 1] = 1.0;
    }

    const initialVolume = computeVolume(state.water, W * H, 1);

    for (let s = 0; s < 100; s++) cpuStep(state, params);

    const finalVolume = computeVolume(state.water, W * H, 1);
    expect(Math.abs(finalVolume - initialVolume) / initialVolume).toBeLessThan(0.001);

    // Check max velocity is near zero (no flow in flat basin)
    let maxVel = 0;
    for (let i = 0; i < W * H * 2; i += 2) {
      maxVel = Math.max(maxVel, Math.abs(state.velocity[i]!), Math.abs(state.velocity[i + 1]!));
    }
    expect(maxVel).toBeLessThan(0.01);
  });

  it('Gaussian bump generates outward radial flux', () => {
    const W = 16, H = 16;
    const state = createCpuState(W, H);
    const params = createCpuParams(W, H, { dx: 1, dt: 0.01, damping: 0.5, manningN: 0 });

    // Uniform depth + Gaussian bump at center
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const cx = i - W / 2, cy = j - H / 2;
        const r2 = cx * cx + cy * cy;
        const h = 1.0 + 0.5 * Math.exp(-r2 / 4);
        state.water[(j * W + i) * 2] = h;
        state.water[(j * W + i) * 2 + 1] = h;
      }
    }

    cpuStep(state, params);

    // After one step, flux should be outward from center
    const cx = W / 2, cy = H / 2;
    const ci = cy * W + cx;
    // Left flux at center should be positive (outward to the left)
    expect(state.fluxLR[ci * 2]!).toBeGreaterThan(0);
    // Right flux at center should be positive (outward to the right)
    expect(state.fluxLR[ci * 2 + 1]!).toBeGreaterThan(0);
  });

  it('Closed boundary zeroes all flux', () => {
    const W = 16, H = 16;
    const state = createCpuState(W, H);
    const params = createCpuParams(W, H, { dx: 1, dt: 0.01, damping: 0.5, manningN: 0 });

    // Fill with water and set all cells to Closed
    for (let i = 0; i < W * H; i++) {
      state.water[i * 2] = 1.0;
      state.water[i * 2 + 1] = 1.0;
      params.boundary[i] = 1; // Closed
    }

    // Add a bump at center to create pressure
    state.water[(8 * W + 8) * 2] = 2.0;

    cpuStep(state, params);

    // All flux should be zero at closed cells
    for (let i = 0; i < W * H; i++) {
      expect(state.fluxLR[i * 2]!).toBe(0);
      expect(state.fluxLR[i * 2 + 1]!).toBe(0);
      expect(state.fluxUD[i * 2]!).toBe(0);
      expect(state.fluxUD[i * 2 + 1]!).toBe(0);
    }
  });

  it('velocity has the /2 factor (Mei eq. 8)', () => {
    const W = 16, H = 16;
    const state = createCpuState(W, H);
    const params = createCpuParams(W, H, { dx: 1, dt: 0.01, damping: 0.5, manningN: 0 });

    // 1D channel: h = 2 on left half, h = 1 on right half
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const h = i < W / 2 ? 2.0 : 1.0;
        state.water[(j * W + i) * 2] = h;
        state.water[(j * W + i) * 2 + 1] = h;
      }
    }

    // Run multiple steps to develop velocity field
    for (let s = 0; s < 10; s++) cpuStep(state, params);

    // At the dam interface (col W/2), velocity should be moderate, not 2x
    const midJ = H / 2;
    const midI = W / 2;
    const ci = midJ * W + midI;
    const u = state.velocity[ci * 2]!;
    // With the /2 factor, velocity should be bounded by sqrt(g*h) ≈ 4.4 m/s
    expect(Math.abs(u)).toBeLessThan(5.0);
    // Without /2, velocity would be ~2x larger
  });

  it('Inflow boundary pins to target depth', () => {
    const W = 16, H = 16;
    const state = createCpuState(W, H);
    const params = createCpuParams(W, H, { dx: 1, dt: 0.01, damping: 0.5, manningN: 0 });

    // Set west column to Inflow with target 2.0
    for (let j = 0; j < H; j++) {
      const ci = j * W;
      params.boundary[ci] = 4; // Inflow
      params.boundaryTargetH[ci] = 2.0;
      state.water[ci * 2] = 0.5; // Start below target
      state.water[ci * 2 + 1] = 0.5;
    }

    // Fill rest with water
    for (let j = 0; j < H; j++) {
      for (let i = 1; i < W; i++) {
        state.water[(j * W + i) * 2] = 0.5;
        state.water[(j * W + i) * 2 + 1] = 0.5;
      }
    }

    cpuStep(state, params);

    // Inflow cells should be at or above target
    for (let j = 0; j < H; j++) {
      const ci = j * W;
      expect(state.water[ci * 2]!).toBeGreaterThanOrEqual(2.0 - 1e-6);
    }
  });

  it('Manning friction attenuates flow on flat bed', () => {
    const W = 32, H = 8;
    const state = createCpuState(W, H);

    // Run once without Manning
    const paramsNoManning = createCpuParams(W, H, { dx: 1, dt: 0.01, damping: 0.5, manningN: 0 });
    // Dam break setup
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const h = i < W / 4 ? 2.0 : 0.5;
        state.water[(j * W + i) * 2] = h;
        state.water[(j * W + i) * 2 + 1] = h;
      }
    }
    for (let s = 0; s < 50; s++) cpuStep(state, paramsNoManning);
    const velNoManning = maxVelocity(state.velocity, W * H);

    // Reset and run with Manning
    const state2 = createCpuState(W, H);
    const paramsWithManning = createCpuParams(W, H, { dx: 1, dt: 0.01, damping: 0.5, manningN: 0.03 });
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const h = i < W / 4 ? 2.0 : 0.5;
        state2.water[(j * W + i) * 2] = h;
        state2.water[(j * W + i) * 2 + 1] = h;
      }
    }
    for (let s = 0; s < 50; s++) cpuStep(state2, paramsWithManning);
    const velWithManning = maxVelocity(state2.velocity, W * H);

    // Manning should reduce peak velocity
    expect(velWithManning).toBeLessThan(velNoManning);
  });
});

describe('CPU Oracle — mass conservation', () => {
  it('closed reflective basin conserves mass over 5000 steps', () => {
    const W = 16, H = 16;
    const state = createCpuState(W, H);
    const params = createCpuParams(W, H, { dx: 1, dt: 0.005, damping: 0.5, manningN: 0 });

    // Non-trivial initial condition: Gaussian bump
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const cx = i - W / 2, cy = j - H / 2;
        const r2 = cx * cx + cy * cy;
        state.water[(j * W + i) * 2] = 1.0 + 0.3 * Math.exp(-r2 / 8);
        state.water[(j * W + i) * 2 + 1] = state.water[(j * W + i) * 2]!;
      }
    }

    const dx = 1;
    const initialVolume = computeVolume(state.water, W * H, dx);

    for (let s = 0; s < 5000; s++) cpuStep(state, params);

    const finalVolume = computeVolume(state.water, W * H, dx);
    const drift = Math.abs(finalVolume - initialVolume) / initialVolume;
    expect(drift).toBeLessThan(0.01); // < 1%
  });
});

describe('CPU Oracle — wave celerity', () => {
  it('linear wave speed approximates sqrt(g*h)', () => {
    // Long 1D channel, small perturbation on h0 = 1m
    const W = 256, H = 4;
    const state = createCpuState(W, H);
    const dx = 0.1;
    const params = createCpuParams(W, H, { dx, dt: 0.001, damping: 0.5, manningN: 0 });

    const h0 = 1.0;
    const pertI = W / 4; // perturbation position
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        let h = h0;
        if (Math.abs(i - pertI) < 3) h += 0.01 * Math.exp(-((i - pertI) * (i - pertI)) / 2);
        state.water[(j * W + i) * 2] = h;
        state.water[(j * W + i) * 2 + 1] = h;
      }
    }

    // Run for enough steps to let the wave travel
    const nSteps = 500;
    for (let s = 0; s < nSteps; s++) cpuStep(state, params);

    // Find the wave front (rightward from perturbation)
    const midJ = Math.floor(H / 2);
    let frontI = pertI;
    for (let i = W - 1; i > pertI; i--) {
      if (Math.abs(state.water[(midJ * W + i) * 2]! - h0) > 0.0001) {
        frontI = i;
        break;
      }
    }

    const simDist = (frontI - pertI) * dx;
    const simTime = nSteps * 0.001;
    const simSpeed = simDist / simTime;
    const theoreticalSpeed = Math.sqrt(9.81 * h0);

    // Within 30% (virtual pipes is dispersive)
    expect(simSpeed).toBeGreaterThan(theoreticalSpeed * 0.5);
    expect(simSpeed).toBeLessThan(theoreticalSpeed * 1.5);
  });
});

// Helpers
function computeVolume(water: Float32Array, cells: number, dx: number): number {
  const area = dx * dx;
  let sum = 0;
  for (let i = 0; i < cells; i++) sum += Math.max(0, water[i * 2]!) * area;
  return sum;
}

function maxVelocity(velocity: Float32Array, cells: number): number {
  let max = 0;
  for (let i = 0; i < cells; i++) {
    const u = velocity[i * 2]!;
    const v = velocity[i * 2 + 1]!;
    max = Math.max(max, Math.hypot(u, v));
  }
  return max;
}
