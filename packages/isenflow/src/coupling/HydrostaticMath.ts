/**
 * Pure functions implementing the water → body force formulas from spec §9.
 *
 * All quantities SI. No GPU, no three.js, no Rapier imports — testable from Node.
 */

export const WATER_DENSITY = 1000; // kg/m³
export const GRAVITY = 9.81; // m/s²

/**
 * Buoyancy force on a single cell.
 *
 * F = ρ · g · V_submerged_cell    (upward, +Y)
 *
 * V_submerged_cell = dx² · min(h, body_top_y − cell_floor_y)
 */
export function cellBuoyancy(
  h: number,
  bodyTopY: number,
  cellFloorY: number,
  dx: number,
  rho: number = WATER_DENSITY,
  g: number = GRAVITY,
): number {
  if (h <= 0) return 0;
  const submergedDepth = Math.min(h, bodyTopY - cellFloorY);
  if (submergedDepth <= 0) return 0;
  return rho * g * dx * dx * submergedDepth;
}

/**
 * Hydrostatic horizontal force on a wall cell, derived from
 * depth-squared pressure integrated over the wall face.
 *
 * F_x = ½ · ρ · g · (h_west² − h_east²) · dx
 */
export function hydrostaticHorizontalForce(
  hWest: number,
  hEast: number,
  dx: number,
  rho: number = WATER_DENSITY,
  g: number = GRAVITY,
): number {
  const hw = Math.max(0, hWest);
  const he = Math.max(0, hEast);
  return 0.5 * rho * g * (hw * hw - he * he) * dx;
}

/**
 * Quadratic drag force on a body cell, in the plane.
 *
 * F = ½ · Cd · ρ · h · |v_rel| · v_rel · dx
 *
 * v_rel = (water_velocity) − (body_velocity_at_cell)
 */
export function cellDragForce(
  h: number,
  waterVx: number,
  waterVz: number,
  bodyVx: number,
  bodyVz: number,
  dx: number,
  cd = 1.0,
  rho: number = WATER_DENSITY,
): readonly [number, number] {
  if (h <= 0) return [0, 0];
  const rx = waterVx - bodyVx;
  const rz = waterVz - bodyVz;
  const speed = Math.hypot(rx, rz);
  if (speed === 0) return [0, 0];
  const k = 0.5 * cd * rho * h * speed * dx;
  return [k * rx, k * rz];
}

/**
 * Vertical damping force, applied when a body is sinking through water.
 * F = -k_v · submerged_fraction · v_body_y · dx²
 *
 * Returns a value that is zero or opposes vertical motion (no buoyancy here).
 */
export function verticalDamping(
  bodyVy: number,
  submergedFraction: number,
  dx: number,
  kV = 5000,
): number {
  if (submergedFraction <= 0) return 0;
  return -kV * submergedFraction * bodyVy * dx * dx;
}

/**
 * The classic dam-break analytical wave-front position.
 *
 * For an instantaneous collapse of a column of height H₀ over a dry flat bed,
 * the leading wave front travels at speed 2·√(g·H₀):
 *
 *     x_front(t) = 2 · √(g · H₀) · t
 *
 * (Stoker, 1957. Used by Vitest dam-break smoke test as the ground truth.)
 */
export function damBreakFrontPosition(initialDepth: number, time: number, g: number = GRAVITY): number {
  return 2 * Math.sqrt(g * initialDepth) * time;
}

/** Cross product of (r × F) for torque accumulation. */
export function cross3(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
): readonly [number, number, number] {
  return [
    ay * bz - az * by,
    az * bx - ax * bz,
    ax * by - ay * bx,
  ];
}
