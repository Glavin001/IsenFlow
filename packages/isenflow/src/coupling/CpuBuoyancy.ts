/**
 * CPU-side buoyancy + vertical damping, computed from the body's CURRENT
 * position each frame with zero latency. This eliminates the oscillation
 * caused by the 3-frame GPU readback delay in ForceReadback.
 */
import { WATER_DENSITY, GRAVITY } from './HydrostaticMath.js';

export interface BodyBuoyancyInput {
  comY: number;
  halfY: number;
  linvelY: number;
  waterLevel: number; // bed + water depth at body position
  bedLevel: number;   // terrain elevation at body position
  footprintCells: number;
  dx: number;
}

export interface BuoyancyResult {
  fy: number;
  submergedFraction: number;
}

/**
 * Compute buoyancy force on a body from its current state.
 * Mirrors the GPU shader (accumulate_forces.wgsl lines 69-79).
 */
export function computeCpuBuoyancy(input: BodyBuoyancyInput, rho = WATER_DENSITY, g = GRAVITY): BuoyancyResult {
  const { comY, halfY, waterLevel, bedLevel, footprintCells, dx } = input;

  const bodyTop = comY + halfY;
  const bodyBot = comY - halfY;
  const topInWater = Math.min(waterLevel, bodyTop);
  const botInWater = Math.max(bedLevel, bodyBot);
  const submergedH = Math.max(0, topInWater - botInWater);

  const bodyHTotal = Math.max(1e-3, 2 * halfY);
  const submergedFraction = Math.min(1, Math.max(0, submergedH / bodyHTotal));

  // Force per cell is rho * g * dx^2 * submergedH; sum over footprint cells
  const fy = rho * g * dx * dx * submergedH * footprintCells;

  return { fy, submergedFraction };
}

/**
 * Vertical damping force opposing vertical motion when submerged.
 * Mirrors the GPU shader (accumulate_forces.wgsl lines 81-88).
 */
export function computeVerticalDamping(
  bodyVy: number,
  submergedFraction: number,
  dx: number,
  footprintCells: number,
  kv = 5000,
): number {
  if (submergedFraction <= 0) return 0;
  return -kv * submergedFraction * bodyVy * dx * dx * footprintCells;
}

/** Estimate number of grid cells a body occupies in the XZ plane. */
export function estimateFootprintCells(halfX: number, halfZ: number, dx: number): number {
  return Math.max(1, Math.ceil((2 * halfX) / dx) * Math.ceil((2 * halfZ) / dx));
}
