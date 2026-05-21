/**
 * SWASHES analytic solutions for shallow-water benchmarking.
 *
 * Pure-TS implementations of four representative solutions from:
 *   Delestre, Lucas, Ksinant, Darboux, Laguerre, Vo, James, Cordier (2013),
 *   "SWASHES: a compilation of Shallow Water Analytic Solutions for Hydraulic
 *   and Environmental Studies", https://www.idpoisson.fr/en/swashes/
 *
 * Each function returns analytic h(x,t) and/or u(x,t) for validation against
 * the GPU solver output.
 */

/**
 * Stoker/Ritter dam-break on a dry bed (SWASHES §3.1.1).
 *
 * Ref: Stoker, J.J. (1957), "Water Waves", Wiley-Interscience.
 *      Ritter, A. (1892), "Die Fortpflanzung der Wasserwellen", VDI.
 *
 * Initial condition: h = H0 for x < 0, h = 0 for x ≥ 0.
 * At time t > 0, the solution consists of a rarefaction fan and a dry front.
 *
 * @param H0 Initial upstream depth (m)
 * @param t  Time (s), must be > 0
 * @param x  Position relative to dam (m), negative = upstream
 * @param g  Gravity (m/s²), default 9.81
 * @returns { h, u } — water depth and velocity at (x, t)
 */
export function stokerDamBreak(
  H0: number,
  t: number,
  x: number,
  g = 9.81,
): { h: number; u: number } {
  if (t <= 0) {
    return { h: x < 0 ? H0 : 0, u: 0 };
  }
  const c0 = Math.sqrt(g * H0);
  const xi = x / t;

  // Three regions:
  // 1. Undisturbed upstream: x/t < -c0
  if (xi < -c0) {
    return { h: H0, u: 0 };
  }
  // 2. Rarefaction fan: -c0 ≤ x/t ≤ 2c0
  if (xi <= 2 * c0) {
    const h = (1 / (9 * g)) * (2 * c0 - xi) ** 2;
    const u = (2 / 3) * (xi + c0);
    return { h, u };
  }
  // 3. Dry bed: x/t > 2c0
  return { h: 0, u: 0 };
}

/**
 * Position of the dam-break dry front at time t.
 *
 * @param H0 Initial upstream depth (m)
 * @param t  Time (s)
 * @param g  Gravity (m/s²)
 * @returns Front position (m) relative to dam
 */
export function stokerFrontPosition(H0: number, t: number, g = 9.81): number {
  return 2 * Math.sqrt(g * H0) * t;
}

/**
 * Thacker parabolic basin oscillation (SWASHES §4.2.2).
 *
 * Ref: Thacker, W.C. (1981), "Some exact solutions to the nonlinear
 *      shallow-water wave equations", J. Fluid Mech., 107, pp. 499-508.
 *
 * Parabolic bowl bed: b(r) = h0 * r² / a²
 * The free surface oscillates with period T = 2π / ω, where ω = √(2gh0) / a.
 *
 * @param a   Basin half-width (m)
 * @param h0  Depth scale (m): b(a) = h0
 * @param t   Time (s)
 * @param r   Radial distance from center (m)
 * @param g   Gravity (m/s²)
 * @returns { h, period } — water depth and oscillation period
 */
export function thackerParabolicBasin(
  a: number,
  h0: number,
  t: number,
  r: number,
  g = 9.81,
): { h: number; period: number } {
  const omega = Math.sqrt(2 * g * h0) / a;
  const period = (2 * Math.PI) / omega;

  // Thacker's solution: eta(r,t) = h0 * (2*A*cos(omega*t) - A² - r²/a²) / (1 - A²)
  // where A is the amplitude parameter. For a simple sloshing mode, A = 0.5 (half-amplitude).
  const A = 0.5;
  const eta = h0 * (2 * A * Math.cos(omega * t) - A * A - (r * r) / (a * a)) / (1 - A * A);
  const bed = h0 * (r * r) / (a * a);
  const h = Math.max(0, eta - bed);

  return { h, period };
}

/**
 * Lake-at-rest over arbitrary bed (SWASHES §2.1.1).
 *
 * The trivial but critical well-balancedness test: if the free surface is flat
 * (η = η₀) and velocity is zero, the solution should remain stationary for
 * all time.
 *
 * @param zb     Bed elevation at position x
 * @param eta0   Reference free-surface elevation (default 1.0)
 * @returns { h } — water depth (= η₀ - zb, clamped to ≥ 0)
 */
export function lakeAtRestOverBump(zb: number, eta0 = 1.0): { h: number } {
  return { h: Math.max(0, eta0 - zb) };
}

/**
 * Manning normal-depth for steady uniform flow on a slope (SWASHES §3.2.1).
 *
 * Ref: MacDonald, I. (1996), extended by Delestre et al. (2013).
 * Chézy/Manning formula for steady uniform flow:
 *   h_n = (n · q / √S₀)^(3/5)
 *   u_n = q / h_n
 *
 * @param q     Unit discharge (m²/s)
 * @param slope Bed slope S₀ (dimensionless, positive for downhill)
 * @param n     Manning roughness coefficient (default 0.03)
 * @returns { h, u } — normal depth and velocity
 */
export function manningNormalDepth(
  q: number,
  slope: number,
  n = 0.03,
): { h: number; u: number } {
  if (slope <= 0 || q <= 0 || n <= 0) {
    return { h: 0, u: 0 };
  }
  const h = Math.pow((n * q) / Math.sqrt(slope), 3 / 5);
  const u = q / h;
  return { h, u };
}
