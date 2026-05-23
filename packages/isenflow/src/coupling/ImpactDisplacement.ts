/**
 * Water-surface impact displacement — reusable API for simulating the effect
 * of falling objects striking a body of water.
 *
 * Instead of using the bed-piston pipeline (which has a 50%-per-frame cap),
 * this module directly writes a crater-and-ring water height pattern that the
 * SWE solver then propagates naturally.
 *
 * Pure functions + composition — no GPU or Three.js dependencies.
 */

import { GRAVITY } from './HydrostaticMath.js';
import type { SplashParticleSystem } from '../particles/SplashParticleSystem.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ImpactShape = 'sphere' | 'cuboid';

export interface ImpactParams {
  /** World-space X position (horizontal). */
  worldX: number;
  /** World-space Z position (horizontal). */
  worldZ: number;
  /** Impact velocity magnitude (m/s, positive = downward). */
  velocity: number;
  /** Object shape. */
  shape: ImpactShape;
  /**
   * Sphere: radius (number).
   * Cuboid: half-extents [hx, hy, hz] where Y is vertical.
   */
  size: number | readonly [number, number, number];
  /** Object density (kg/m³). */
  density: number;
  /** Resting water depth at impact location (m). */
  waterDepth: number;
}

export interface ImpactResult {
  /** Grid region that was modified. */
  region: { x: number; y: number; w: number; h: number };
  /** Peak ring wave height above resting depth (m). */
  ringHeight: number;
  /** Crater depression depth below resting depth (m). */
  craterDepth: number;
  /** Froude number of impact. */
  froude: number;
}

export interface ComputedImpact extends ImpactResult {
  /** Absolute water depth values for the region (row-major). */
  values: Float32Array;
  /**
   * Per-cell outward radial momentum (hu, hv) for the region (row-major,
   * 2 floats per cell — hu then hv).  When applied to a momentum-aware
   * solver (KP), this drives the wave outward physically rather than
   * relying on h alone to do the work.  VP-compatible callers can ignore
   * this field.
   */
  momentum: Float32Array;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Horizontal cross-section area of the impacting object (m²). */
function crossSectionArea(shape: ImpactShape, size: number | readonly [number, number, number]): number {
  if (shape === 'sphere') {
    const r = size as number;
    return Math.PI * r * r;
  }
  const [hx, , hz] = size as readonly [number, number, number];
  return 4 * hx * hz;
}

/** Vertical half-extent of the object (m). */
function halfHeight(shape: ImpactShape, size: number | readonly [number, number, number]): number {
  if (shape === 'sphere') return size as number;
  return (size as readonly [number, number, number])[1];
}

/** Effective radius of the cross-section (m) — for ring geometry. */
function effectiveRadius(area: number): number {
  return Math.sqrt(area / Math.PI);
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

/**
 * Compute the crater-and-ring displacement pattern for a water impact.
 *
 * @param params  Impact parameters (position, velocity, shape, etc.)
 * @param grid    Grid info: `{ width, height, dx, origin }`.
 * @returns Computed pattern with absolute water depth values and metadata.
 */
export function computeImpact(
  params: ImpactParams,
  grid: { width: number; height: number; dx: number; origin: readonly [number, number] },
): ComputedImpact {
  const { velocity, shape, size, density, waterDepth } = params;
  const dx = grid.dx;

  const csArea = crossSectionArea(shape, size);
  const hY = halfHeight(shape, size);

  // Froude number: dimensionless impact intensity
  const fr = velocity / Math.sqrt(GRAVITY * Math.max(0.01, waterDepth));

  // Volume displaced: object plunges to min(full height, water depth)
  const plungeDepth = Math.min(2 * hY, waterDepth);
  const displacedVolume = csArea * plungeDepth;

  // Density ratio amplifies heavier objects (they plunge deeper/faster)
  const densityRatio = Math.min(3, density / 1000);

  // Crater radius in cells (≈ object cross-section radius)
  const rObj = effectiveRadius(csArea);
  const craterCells = Math.max(2, Math.ceil(rObj / dx));

  // Ring radius expands with Froude number
  const ringCells = Math.max(craterCells + 3, Math.ceil(rObj * (1 + 0.5 * fr) / dx) + 3);

  // Ring area (annular, in m²)
  const ringAreaM2 = Math.PI * ((ringCells * dx) ** 2 - (craterCells * dx) ** 2);

  // Ring height: displaced volume spread over ring area, amplified by Fr and density.
  // Cap at 1.5 × waterDepth — visually impressive but well below VP's 2×
  // band-aid (which contributed to spike artifacts).  KP solver propagates
  // the ring + the radial momentum we inject (see `momentum` field below)
  // as an accurate wave train, so the visual relies less on a tall instant
  // bump.
  const rawRingH = (displacedVolume / Math.max(0.01, ringAreaM2)) * (1 + fr) * densityRatio;
  const ringHeight = Math.min(waterDepth * 1.5, rawRingH);

  // Crater depression: proportional to ring height, capped at 0.7 × waterDepth
  // so we never write h ≤ 0 (which would force the cell dry and lose mass).
  const craterDepth = Math.min(waterDepth * 0.7, ringHeight * 1.2);

  // Build the grid region
  const ci = Math.round((params.worldX - grid.origin[0]) / dx);
  const cj = Math.round((params.worldZ - grid.origin[1]) / dx);

  const margin = ringCells + 1;
  const x0 = Math.max(0, ci - margin);
  const y0 = Math.max(0, cj - margin);
  const x1 = Math.min(grid.width, ci + margin + 1);
  const y1 = Math.min(grid.height, cj + margin + 1);
  const w = x1 - x0;
  const h = y1 - y0;

  const values = new Float32Array(w * h);
  const momentum = new Float32Array(w * h * 2);

  // Reference momentum magnitude: enough to give the ring water an outward
  // velocity comparable to √(g · ringHeight) — the natural shallow-water
  // wave speed at the perturbation amplitude.  This injects "real" kinetic
  // energy that the KP solver propagates radially as a true wave.
  const refSpeed = Math.sqrt(GRAVITY * Math.max(0.01, ringHeight));
  const momentumScale = refSpeed * (waterDepth + ringHeight) * Math.min(2, fr);

  for (let dj = 0; dj < h; dj++) {
    for (let di = 0; di < w; di++) {
      const gx = (x0 + di) - ci;
      const gy = (y0 + dj) - cj;
      const r = Math.hypot(gx, gy); // in cells

      let depth = waterDepth;
      let radialMomentum = 0;

      if (r < craterCells) {
        // Crater zone: cosine-bell depression, slight INWARD momentum
        // (water collapsing into the crater)
        const t = r / Math.max(1, craterCells);
        depth = Math.max(0.01, waterDepth - craterDepth * 0.5 * (1 + Math.cos(Math.PI * t)));
        radialMomentum = -0.2 * momentumScale * Math.sin(Math.PI * t);
      } else if (r < ringCells) {
        // Ring zone: cosine-bell elevation + OUTWARD momentum
        const t = (r - craterCells) / Math.max(1, ringCells - craterCells);
        depth = waterDepth + ringHeight * 0.5 * (1 + Math.cos(Math.PI * t));
        radialMomentum = momentumScale * 0.5 * (1 + Math.cos(Math.PI * t));
      }

      values[dj * w + di] = depth;

      // Convert radial momentum to (hu, hv) components
      const dist = Math.max(1e-6, r);
      const dirX = gx / dist;
      const dirY = gy / dist;
      momentum[(dj * w + di) * 2 + 0] = radialMomentum * dirX;
      momentum[(dj * w + di) * 2 + 1] = radialMomentum * dirY;
    }
  }

  return {
    region: { x: x0, y: y0, w, h },
    values,
    momentum,
    ringHeight,
    craterDepth,
    froude: fr,
  };
}

// ---------------------------------------------------------------------------
// Convenience: apply to solver
// ---------------------------------------------------------------------------

/** Minimal solver interface (avoids importing the full class). */
interface SolverLike {
  readonly grid: { width: number; height: number; dx: number; origin: readonly [number, number] };
  writeWaterRegion(region: { x: number; y: number; w: number; h: number }, values: Float32Array): void;
  /**
   * Optional: KP-aware solvers also accept momentum injection so impacts
   * radiate as true waves rather than relying on h alone.  VP solvers
   * omit this method; we fall back to height-only injection.
   *
   * `state` is packed (h, hu, hv, _pad) per cell (length = region.w·region.h·4).
   */
  writeStateRegion?(
    region: { x: number; y: number; w: number; h: number },
    state: Float32Array,
  ): void;
}

/**
 * Apply an impact to the water surface in one shot.
 * The SWE solver propagates the resulting crater + ring as waves.
 * On KP solvers (with `writeStateRegion`), the impact also delivers
 * accurate outward radial momentum so the wave radiates physically.
 */
export function applyImpact(solver: SolverLike, params: ImpactParams): ImpactResult {
  const computed = computeImpact(params, solver.grid);
  if (solver.writeStateRegion) {
    // KP path: write (h, hu, hv, 0) packed per cell — single writeBuffer per row.
    const w = computed.region.w;
    const h = computed.region.h;
    const state = new Float32Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      state[i * 4 + 0] = computed.values[i]!;             // h
      state[i * 4 + 1] = computed.momentum[i * 2 + 0]!;   // hu
      state[i * 4 + 2] = computed.momentum[i * 2 + 1]!;   // hv
      // state[i*4 + 3] = 0
    }
    solver.writeStateRegion(computed.region, state);
    // Also mirror the depth into the legacy `water` buffer for consumers
    // (renderer, accumulate_forces) that read it before the next step's
    // refresh-views pass.
    solver.writeWaterRegion(computed.region, computed.values);
  } else {
    // VP fallback: depth-only injection (no momentum).
    solver.writeWaterRegion(computed.region, computed.values);
  }
  return {
    region: computed.region,
    ringHeight: computed.ringHeight,
    craterDepth: computed.craterDepth,
    froude: computed.froude,
  };
}

// ---------------------------------------------------------------------------
// Optional: decorative splash particles
// ---------------------------------------------------------------------------

/**
 * Spawn decorative splash particles proportional to impact intensity.
 */
export function spawnImpactSplash(
  splashes: SplashParticleSystem,
  params: ImpactParams,
  result: ImpactResult,
): void {
  const intensity = Math.min(1, result.froude / 5);
  const upwardSpeed = Math.min(8, params.velocity * 0.4);
  splashes.spawn({
    position: [params.worldX, params.waterDepth, params.worldZ],
    intensity,
    upwardSpeed,
  });
}
