/**
 * Per-chunk stress accumulator with exponential decay.
 *
 *     stress_new = stress_old · decay + |F| · dt
 *     fracture when stress > budget
 *
 * Transient loads do not fracture; sustained loads do.
 */

export interface StressState {
  /** Current accumulated stress, N·s. */
  accum: number;
  /** Per-step decay factor in (0, 1]; 1 = no decay. */
  decay: number;
  /** Failure threshold in N·s. */
  budget: number;
  /** True once fractured. */
  fractured: boolean;
}

export function createStress(budget: number, decay = 0.95): StressState {
  return { accum: 0, decay, budget, fractured: false };
}

/**
 * Tick the accumulator by one frame.
 * Returns true if this tick caused a fracture (rising edge).
 */
export function tickStress(state: StressState, forceMagnitude: number, dt: number): boolean {
  if (state.fractured) return false;
  state.accum = state.accum * state.decay + forceMagnitude * dt;
  if (state.accum > state.budget) {
    state.fractured = true;
    return true;
  }
  return false;
}
