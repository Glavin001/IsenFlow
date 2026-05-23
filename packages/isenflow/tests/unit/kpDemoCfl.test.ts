/**
 * Verify KP is stable at the dt/dx values the production demos actually use.
 *
 * The other tests run at very conservative CFL (~0.03).  The demos run at
 * CFL ≈ 0.16 (with the 4×-finer dt fix) or 0.65 (the original).  These
 * tests cover the high-CFL regime so any future change that bumps dt is
 * caught by Vitest, not by visual playtesting.
 */
import { describe, it, expect } from 'vitest';
import {
  createKpState,
  createKpParams,
  createKpScratch,
  kpStep,
  BT_CLOSED,
  type KpState,
} from '../../src/core/cpu/CpuKurganov.js';

function maxH(state: KpState): number {
  let m = 0;
  for (let i = 0; i < state.h.length; i++) if (state.h[i]! > m) m = state.h[i]!;
  return m;
}

function hasBadNumbers(state: KpState): boolean {
  for (const arr of [state.h, state.hu, state.hv]) {
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i]!)) return true;
    }
  }
  return false;
}

describe('KP — production-demo CFL regime', () => {
  it('dam-break at demo dx=0.042, dt=1/960 (CFL≈0.16): no spikes for 5 s', () => {
    // Mirror 01-dam-break setup at coarser cell count (32×32 instead of
    // 384×384 to keep the test fast) but same dx, dt as production.
    const W = 64, H = 16;
    const dx = 16 / 384;                  // 0.0417 m (demo grid)
    const dt = 1 / 960;                   // demo dt after CFL fix
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx, dt, gravity: 9.81, manningN: 0.03 });
    for (let c = 0; c < params.boundary.length; c++) params.boundary[c] = BT_CLOSED;

    // Reservoir h=4 on left third, dry on right; tall bed dam at i=W/3
    const fillEnd = Math.floor(W / 3);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        state.h[j * W + i] = i < fillEnd ? 4 : 0;
      }
    }
    for (let j = 0; j < H; j++) state.bed[j * W + fillEnd] = 6;

    const scratch = createKpScratch(W, H);
    // Drop the dam after 0.5 s
    let droppedAt = -1;
    const totalSteps = Math.round(5.0 / dt);
    for (let s = 0; s < totalSteps; s++) {
      if (droppedAt < 0 && s * dt > 0.5) {
        for (let j = 0; j < H; j++) state.bed[j * W + fillEnd] = 0;
        droppedAt = s;
      }
      kpStep(state, params, scratch);
      // Bail out early if anything goes catastrophic
      if (s % 100 === 0) {
        if (hasBadNumbers(state)) throw new Error(`NaN at step ${s} (t=${(s * dt).toFixed(3)}s)`);
        const m = maxH(state);
        if (m > 20) throw new Error(`SPIKE h=${m.toFixed(2)} at step ${s} (t=${(s * dt).toFixed(3)}s)`);
      }
    }

    expect(hasBadNumbers(state)).toBe(false);
    expect(maxH(state)).toBeLessThan(8);  // Initial reservoir h=4, allow 2× max
  });

  it('initial-condition lake-at-rest at demo CFL: |u|, |v| stay small for 500 steps', () => {
    const W = 32, H = 32;
    const dx = 16 / 384;
    const dt = 1 / 960;
    const state = createKpState(W, H);
    const params = createKpParams(W, H, { dx, dt, gravity: 9.81, manningN: 0 });
    for (let c = 0; c < params.boundary.length; c++) params.boundary[c] = BT_CLOSED;
    for (let i = 0; i < state.h.length; i++) state.h[i] = 1.0;

    const scratch = createKpScratch(W, H);
    for (let s = 0; s < 500; s++) kpStep(state, params, scratch);

    let maxMomentum = 0;
    for (let i = 0; i < state.h.length; i++) {
      maxMomentum = Math.max(maxMomentum, Math.abs(state.hu[i]!), Math.abs(state.hv[i]!));
    }
    expect(hasBadNumbers(state)).toBe(false);
    expect(maxMomentum).toBeLessThan(1e-3);
  });
});
