/**
 * CPU reference implementation of the Kurganov–Petrova central-upwind
 * scheme for the 2D Saint-Venant (shallow-water) system.
 *
 * This is the algebraic oracle against which GPU kernels are byte-verified.
 *
 * References
 * ----------
 *  - Kurganov, A. & Petrova, G. (2007). "A second-order well-balanced
 *    positivity-preserving central-upwind scheme for the Saint-Venant system."
 *    Comm. Math. Sci. 5(1): 133–160.
 *  - Audusse, E., Bouchut, F., Bristeau, M-O., Klein, R. & Perthame, B. (2004).
 *    "A fast and stable well-balanced scheme with hydrostatic reconstruction
 *    for shallow water flows." SIAM J. Sci. Comput. 25(6): 2050–2065.
 *  - SWASHES test suite §1: Manning-friction analytic profiles.
 *
 * Conservative state per cell (row-major, idx = j*W + i):
 *   h   : water depth (m)
 *   hu  : x-momentum density (m²/s)   — primary state, NOT reconstructed
 *   hv  : y-momentum density (m²/s)   — primary state, NOT reconstructed
 *   bed : bed elevation B (m, constant between solver steps)
 *
 * Velocity is *derived* from the conserved momentum using the Kurganov
 * desingularization (eqs. 2.16–2.17 in KP07):
 *
 *      u = √2 · h · hu / √(h⁴ + max(h⁴, ε⁴))
 *
 * which is smooth across h → 0 (no `max(0.05, h)` cliff that the old VP
 * scheme suffered from).
 *
 * Spatial discretization: second-order MUSCL with a generalized minmod
 * slope limiter (θ ∈ [1, 2], we use θ = 1.3 for accuracy without
 * overshoot). Wet/dry interfaces use Audusse hydrostatic reconstruction.
 *
 * Time integration: SSP-RK2 (Heun) — two stages per logical step, both
 * stages use the same spatial operator L:
 *
 *      U⁽¹⁾   = Uⁿ + Δt · L(Uⁿ)
 *      Uⁿ⁺¹  = ½ Uⁿ + ½ (U⁽¹⁾ + Δt · L(U⁽¹⁾))
 *
 * Boundary conditions (per cell, encoded in `boundary`):
 *   0 Interior — normal SWE update
 *   1 Closed   — reflective: normal momentum component flips at faces
 *   2 Open     — transmissive: ghost = self
 *   3 Sponge   — damp momentum to zero, damp h toward neighbor mean
 *   4 Inflow   — h pinned to boundaryTargetH[idx], momentum unchanged
 *   5 Sea      — h pinned to boundaryTargetH[idx], momentum zeroed
 *   6 Solid    — cell stays (0, 0, 0); zero flux through faces touching it
 */

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

export interface KpState {
  /** Water depth h, length = width × height. */
  h: Float32Array;
  /** x-momentum hu = h·u, length = width × height. */
  hu: Float32Array;
  /** y-momentum hv = h·v, length = width × height. */
  hv: Float32Array;
  /** Bed elevation B (read-only during a step), length = width × height. */
  bed: Float32Array;
}

export interface KpParams {
  readonly width: number;
  readonly height: number;
  readonly dx: number;
  readonly dt: number;
  readonly gravity: number;
  /** Manning roughness coefficient (default 0; 0.03 for natural channels). */
  readonly manningN: number;
  /**
   * Desingularization regularizer ε for u = √2·h·hu / √(h⁴ + max(h⁴, ε⁴)).
   * Default 1e-3 m.  Smaller = more accurate but more sensitive to noise.
   */
  readonly desingEpsilon: number;
  /** Boundary type per cell. */
  readonly boundary: Uint32Array;
  /** Per-cell target depth used by Inflow (4) and Sea (5). */
  readonly boundaryTargetH: Float32Array;
}

/** Boundary type constants (mirror of `boundaries/BoundaryConditions.ts`). */
export const BT_INTERIOR = 0 as const;
export const BT_CLOSED = 1 as const;
export const BT_OPEN = 2 as const;
export const BT_SPONGE = 3 as const;
export const BT_INFLOW = 4 as const;
export const BT_SEA = 5 as const;
export const BT_SOLID = 6 as const;

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

export function createKpState(width: number, height: number): KpState {
  const cells = width * height;
  return {
    h: new Float32Array(cells),
    hu: new Float32Array(cells),
    hv: new Float32Array(cells),
    bed: new Float32Array(cells),
  };
}

export function createKpParams(
  width: number,
  height: number,
  opts?: Partial<
    Omit<KpParams, 'width' | 'height' | 'boundary' | 'boundaryTargetH'>
  >,
): KpParams {
  const cells = width * height;
  return {
    width,
    height,
    dx: opts?.dx ?? 1,
    dt: opts?.dt ?? 1 / 240,
    gravity: opts?.gravity ?? 9.81,
    manningN: opts?.manningN ?? 0,
    desingEpsilon: opts?.desingEpsilon ?? 1e-3,
    boundary: new Uint32Array(cells),
    boundaryTargetH: new Float32Array(cells),
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SQRT2 = Math.SQRT2;

/** Generalized minmod limiter with parameter θ ∈ [1, 2].  θ=1 → minmod, θ=2 → MC. */
function minmod3(a: number, b: number, c: number): number {
  if (a > 0 && b > 0 && c > 0) return Math.min(a, b, c);
  if (a < 0 && b < 0 && c < 0) return Math.max(a, b, c);
  return 0;
}

/**
 * Kurganov desingularization: smooth approximation of u = hu/h that
 * vanishes smoothly as h → 0.  See KP07 eqs. 2.16–2.17.
 */
export function desingularize(h: number, q: number, eps: number): number {
  if (h <= 0) return 0;
  const h2 = h * h;
  const h4 = h2 * h2;
  const eps4 = eps * eps * eps * eps;
  return (SQRT2 * h * q) / Math.sqrt(h4 + Math.max(h4, eps4));
}

function idx(i: number, j: number, w: number): number {
  return j * w + i;
}

function inBounds(i: number, j: number, w: number, h: number): boolean {
  return i >= 0 && j >= 0 && i < w && j < h;
}

/**
 * Read a cell value or return the "ghost" value implied by the boundary
 * type when the indexed cell is out of bounds (used at the domain edge).
 *
 * For Open we return the self value (transmissive); for everything else
 * (including the default) we return the self value as well — the actual
 * wall/no-flux behaviour is enforced inside the flux computation by
 * checking the neighbor's boundary type, NOT by ghost cells alone.
 */
function readGhost(
  arr: Float32Array,
  i: number,
  j: number,
  w: number,
  h: number,
  selfValue: number,
): number {
  if (inBounds(i, j, w, h)) return arr[idx(i, j, w)]!;
  return selfValue;
}

// ---------------------------------------------------------------------------
// Slope reconstruction
// ---------------------------------------------------------------------------

/**
 * Limited slopes for the *primitive-reconstruction* variables (w, hu, hv).
 *
 * KP07 §2 reconstructs (w, hu, hv) — NOT (h, hu, hv) directly — because
 * limiting on w + cell-piecewise-constant B guarantees the well-balanced
 * C-property: for lake at rest (w = const, hu = hv = 0), all slopes are
 * exactly zero so the scheme reduces to a steady state.
 *
 * h at a face is *derived*:
 *     h_L_face = w_L_pre - B_i_cell-center
 *     h_R_face = w_R_pre - B_neighbor_cell-center
 *
 * Storage: 6 floats per cell (dw_x, dw_y, dhu_x, dhu_y, dhv_x, dhv_y).
 *
 * Dry cells (h ≤ ε_dry) get zero slopes — preserves positivity trivially.
 */
export interface KpSlopes {
  dw_x: Float32Array;
  dw_y: Float32Array;
  dhu_x: Float32Array;
  dhu_y: Float32Array;
  dhv_x: Float32Array;
  dhv_y: Float32Array;
}

export function createKpSlopes(width: number, height: number): KpSlopes {
  const cells = width * height;
  return {
    dw_x: new Float32Array(cells),
    dw_y: new Float32Array(cells),
    dhu_x: new Float32Array(cells),
    dhu_y: new Float32Array(cells),
    dhv_x: new Float32Array(cells),
    dhv_y: new Float32Array(cells),
  };
}

const SLOPE_THETA = 1.3;
const DRY_THRESHOLD = 1e-5;

export function computeKpSlopes(
  state: KpState,
  params: KpParams,
  out: KpSlopes,
): void {
  const { width: W, height: H } = params;
  const { h, hu, hv, bed } = state;
  const theta = SLOPE_THETA;

  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const c = idx(i, j, W);
      const hC = h[c]!;

      // Dry cell — no reconstruction (slopes set to zero).
      if (hC <= DRY_THRESHOLD) {
        out.dw_x[c] = 0;
        out.dw_y[c] = 0;
        out.dhu_x[c] = 0;
        out.dhu_y[c] = 0;
        out.dhv_x[c] = 0;
        out.dhv_y[c] = 0;
        continue;
      }

      const wC = hC + bed[c]!;

      // X-direction slopes (need i-1, i, i+1)
      if (i > 0 && i < W - 1) {
        const cL = c - 1;
        const cR = c + 1;
        const wL = h[cL]! + bed[cL]!;
        const wR = h[cR]! + bed[cR]!;

        out.dw_x[c] = minmod3(
          theta * (wC - wL),
          0.5 * (wR - wL),
          theta * (wR - wC),
        );
        out.dhu_x[c] = minmod3(
          theta * (hu[c]! - hu[cL]!),
          0.5 * (hu[cR]! - hu[cL]!),
          theta * (hu[cR]! - hu[c]!),
        );
        out.dhv_x[c] = minmod3(
          theta * (hv[c]! - hv[cL]!),
          0.5 * (hv[cR]! - hv[cL]!),
          theta * (hv[cR]! - hv[c]!),
        );
      } else {
        out.dw_x[c] = 0;
        out.dhu_x[c] = 0;
        out.dhv_x[c] = 0;
      }

      // Y-direction slopes (need j-1, j, j+1)
      if (j > 0 && j < H - 1) {
        const cD = c - W;
        const cU = c + W;
        const wD = h[cD]! + bed[cD]!;
        const wU = h[cU]! + bed[cU]!;

        out.dw_y[c] = minmod3(
          theta * (wC - wD),
          0.5 * (wU - wD),
          theta * (wU - wC),
        );
        out.dhu_y[c] = minmod3(
          theta * (hu[c]! - hu[cD]!),
          0.5 * (hu[cU]! - hu[cD]!),
          theta * (hu[cU]! - hu[c]!),
        );
        out.dhv_y[c] = minmod3(
          theta * (hv[c]! - hv[cD]!),
          0.5 * (hv[cU]! - hv[cD]!),
          theta * (hv[cU]! - hv[c]!),
        );
      } else {
        out.dw_y[c] = 0;
        out.dhu_y[c] = 0;
        out.dhv_y[c] = 0;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Central-upwind flux at a single face (Audusse-reconstructed)
// ---------------------------------------------------------------------------

/**
 * Compute the central-upwind flux at a face, given the two-sided
 * reconstructed conservative states.  Returns the flux components
 * (F^h, F^hu, F^hv) in the *normal* direction.
 *
 * @param hL, huL, hvL  reconstructed conservative state on the LEFT  (or DOWN) side
 * @param hR, huR, hvR  reconstructed conservative state on the RIGHT (or UP)   side
 * @param normalAxis    0 for x-faces, 1 for y-faces (decides which momentum
 *                      component is "normal" vs "tangent")
 * @param g             gravity
 * @param eps           desingularization ε
 * @param out           length-3 array where flux components are written
 *                      [Fh, Fhu, Fhv]
 *
 * Also returns the local maximum wave speed |a±| via the .maxSpeed property
 * of `out` (using a workaround: encode in out[3] if length >= 4).
 *
 * Returns the local max wave speed (for CFL computation).
 */
function centralUpwindFlux(
  hL: number,
  huL: number,
  hvL: number,
  hR: number,
  huR: number,
  hvR: number,
  normalAxis: 0 | 1,
  g: number,
  eps: number,
  out: { h: number; hu: number; hv: number },
): number {
  // Desingularized normal velocity on each side
  const uL = desingularize(hL, normalAxis === 0 ? huL : hvL, eps);
  const uR = desingularize(hR, normalAxis === 0 ? huR : hvR, eps);

  const cL = Math.sqrt(g * Math.max(0, hL));
  const cR = Math.sqrt(g * Math.max(0, hR));

  // One-sided wave speeds (KP07 eq. 2.13–2.14)
  const aPlus = Math.max(uL + cL, uR + cR, 0);
  const aMinus = Math.min(uL - cL, uR - cR, 0);

  const denom = aPlus - aMinus;
  if (denom < 1e-12) {
    // Both speeds ~zero — no flux, no spurious division
    out.h = 0;
    out.hu = 0;
    out.hv = 0;
    return 0;
  }

  // Physical fluxes F(U) for SWE.  When the face normal is x:
  //   F^h  = hu                              ( = q_normal)
  //   F^hu = hu·u + g·h²/2                   (pressure on x-momentum)
  //   F^hv = hu·v                            (advection of v by u)
  // For y-normal faces, swap u↔v and hu↔hv in the same pattern.
  let F_L_h: number, F_L_hu: number, F_L_hv: number;
  let F_R_h: number, F_R_hu: number, F_R_hv: number;
  const pressureL = 0.5 * g * hL * hL;
  const pressureR = 0.5 * g * hR * hR;

  if (normalAxis === 0) {
    // Normal = x.  Use uL,uR as the normal velocity.
    F_L_h = huL;
    F_L_hu = huL * uL + pressureL;
    F_L_hv = huL * desingularize(hL, hvL, eps); // hu · v
    F_R_h = huR;
    F_R_hu = huR * uR + pressureR;
    F_R_hv = huR * desingularize(hR, hvR, eps);
  } else {
    // Normal = y.  Use uL,uR (here = vL,vR) as the normal velocity.
    F_L_h = hvL;
    F_L_hu = hvL * desingularize(hL, huL, eps); // hv · u
    F_L_hv = hvL * uL + pressureL;
    F_R_h = hvR;
    F_R_hu = hvR * desingularize(hR, huR, eps);
    F_R_hv = hvR * uR + pressureR;
  }

  // Central-upwind flux (KP07 eq. 2.11)
  out.h =
    (aPlus * F_L_h - aMinus * F_R_h + aPlus * aMinus * (hR - hL)) / denom;
  out.hu =
    (aPlus * F_L_hu - aMinus * F_R_hu + aPlus * aMinus * (huR - huL)) / denom;
  out.hv =
    (aPlus * F_L_hv - aMinus * F_R_hv + aPlus * aMinus * (hvR - hvL)) / denom;

  return Math.max(aPlus, -aMinus);
}

// ---------------------------------------------------------------------------
// One RK stage (the L operator + Δt scaling, applied as: ΔU = Δt · L(U))
// ---------------------------------------------------------------------------

/**
 * Spatial operator L applied to `state`, scaled by dt, written into `dU_h`,
 * `dU_hu`, `dU_hv`.  Returns the maximum local wave speed observed during
 * flux evaluation (for adaptive CFL accounting).
 *
 * Implements:
 *   - Slope reconstruction (computeKpSlopes)
 *   - Per-face Audusse hydrostatic reconstruction (b★ = max(bed_L, bed_R))
 *   - Central-upwind flux
 *   - Well-balanced source term: S = -g · h · ∇B  (computed at cell centers
 *     using the reconstructed b★ values to preserve C-property)
 *
 * The caller is responsible for: zero-initializing the dU buffers, applying
 * the result (Uⁿ⁺¹ = a·Uⁿ + b·(U + dU)), Manning friction, and boundary
 * conditions.
 */
export function kpSpatialOperator(
  state: KpState,
  params: KpParams,
  slopes: KpSlopes,
  scratchFlux: { h: number; hu: number; hv: number },
  dU_h: Float32Array,
  dU_hu: Float32Array,
  dU_hv: Float32Array,
): number {
  const { width: W, height: H, dx, dt, gravity: g, desingEpsilon: eps, boundary } = params;
  const { h, hu, hv, bed } = state;

  // 1. Slope reconstruction (writes into `slopes`)
  computeKpSlopes(state, params, slopes);

  // 2. Initialize dU to zero
  dU_h.fill(0);
  dU_hu.fill(0);
  dU_hv.fill(0);

  let maxSpeed = 0;
  const dtOverDx = dt / dx;

  // 3. X-faces — between cell (i, j) and (i+1, j) for i in [0..W-1)
  //    We compute each face's flux once and apply it to both adjacent cells.
  //
  //    Reconstructed values:
  //      East-edge of cell (i, j):   U⁺_{i,j} = U_{i,j} + 0.5 · slope_x_{i,j}
  //      West-edge of cell (i+1, j): U⁻_{i+1,j} = U_{i+1,j} - 0.5 · slope_x_{i+1,j}
  //
  //    Audusse hydrostatic reconstruction at the face:
  //      b★ = max(b_E_{i,j}, b_W_{i+1,j})  where b_E_{i,j} = bed_{i,j} + 0.5·dw_x[c]-0.5·dh_x[c]
  //      h*_L = max(0, w⁺_{i,j} - b★)
  //      h*_R = max(0, w⁻_{i+1,j} - b★)
  //
  //    Momentum reconstruction is straightforward:
  //      hu*_L = hu⁺_{i,j} · (h*_L / max(ε, h⁺_{i,j}))   (rescale to keep u steady)
  //      (and similarly for R, hv)
  //
  //    For positivity-preserving reconstruction, this momentum rescaling
  //    is crucial — see Audusse 2004 §3.
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W - 1; i++) {
      const cL = idx(i, j, W);
      const cR = idx(i + 1, j, W);

      // Skip if either side is a Solid cell — zero flux through that face
      const btL = boundary[cL]!;
      const btR = boundary[cR]!;
      if (btL === BT_SOLID || btR === BT_SOLID) continue;

      // Reconstructed primitive surface elevation and momentum at the face
      //   from each side.  Only (w, hu, hv) are slope-limited; h is derived
      //   from w - B (with B piecewise-constant per cell).
      const wLpre = h[cL]! + bed[cL]! + 0.5 * slopes.dw_x[cL]!;
      const wRpre = h[cR]! + bed[cR]! - 0.5 * slopes.dw_x[cR]!;
      const huLpre = hu[cL]! + 0.5 * slopes.dhu_x[cL]!;
      const huRpre = hu[cR]! - 0.5 * slopes.dhu_x[cR]!;
      const hvLpre = hv[cL]! + 0.5 * slopes.dhv_x[cL]!;
      const hvRpre = hv[cR]! - 0.5 * slopes.dhv_x[cR]!;

      // Bed at face from each cell-center (B is piecewise constant in the
      // Audusse formulation — no bed-slope reconstruction).
      const bL = bed[cL]!;
      const bR = bed[cR]!;
      const bStar = Math.max(bL, bR);

      // Pre-Audusse depths (used for momentum rescaling)
      const hLpre = Math.max(0, wLpre - bL);
      const hRpre = Math.max(0, wRpre - bR);

      // Audusse hydrostatic reconstruction (clip at zero depth at the face)
      const hStarL = Math.max(0, wLpre - bStar);
      const hStarR = Math.max(0, wRpre - bStar);

      // Momentum is rescaled by the depth-ratio so velocity at the face is
      // preserved across the reconstruction discontinuity (Audusse §3).
      const uLpre = desingularize(hLpre, huLpre, eps);
      const vLpre = desingularize(hLpre, hvLpre, eps);
      const uRpre = desingularize(hRpre, huRpre, eps);
      const vRpre = desingularize(hRpre, hvRpre, eps);
      const huStarL = hStarL * uLpre;
      const hvStarL = hStarL * vLpre;
      const huStarR = hStarR * uRpre;
      const hvStarR = hStarR * vRpre;

      const localMax = centralUpwindFlux(
        hStarL, huStarL, hvStarL,
        hStarR, huStarR, hvStarR,
        0, g, eps, scratchFlux,
      );
      if (localMax > maxSpeed) maxSpeed = localMax;

      // Conservative update contribution (face area / cell area = 1/dx)
      //   ∂U/∂t = -(F_E - F_W)/dx
      // For left cell:  -F_face / dx  (face is the EAST face)
      // For right cell: +F_face / dx  (face is the WEST face)
      dU_h[cL]! -= dtOverDx * scratchFlux.h;
      dU_hu[cL]! -= dtOverDx * scratchFlux.hu;
      dU_hv[cL]! -= dtOverDx * scratchFlux.hv;
      dU_h[cR]! += dtOverDx * scratchFlux.h;
      dU_hu[cR]! += dtOverDx * scratchFlux.hu;
      dU_hv[cR]! += dtOverDx * scratchFlux.hv;

      // Well-balanced bed-pressure source term.  For each face, the cell on
      // each side gets a g/2 · (h*)² contribution that exactly cancels the
      // pressure component of the flux for lake-at-rest (h*_L = h*_R = h*,
      // u = 0) — guaranteeing the C-property.  Derived in
      //   Audusse 2004 §3.3 and Kurganov-Petrova 2007 §2.4.
      const pressLstar = 0.5 * g * hStarL * hStarL;
      const pressRstar = 0.5 * g * hStarR * hStarR;
      dU_hu[cL]! += dtOverDx * pressLstar;
      dU_hu[cR]! -= dtOverDx * pressRstar;
    }
  }

  // 4. Y-faces — between cell (i, j) and (i, j+1) for j in [0..H-1)
  for (let j = 0; j < H - 1; j++) {
    for (let i = 0; i < W; i++) {
      const cD = idx(i, j, W);
      const cU = idx(i, j + 1, W);

      const btD = boundary[cD]!;
      const btU = boundary[cU]!;
      if (btD === BT_SOLID || btU === BT_SOLID) continue;

      const wDpre = h[cD]! + bed[cD]! + 0.5 * slopes.dw_y[cD]!;
      const wUpre = h[cU]! + bed[cU]! - 0.5 * slopes.dw_y[cU]!;
      const huDpre = hu[cD]! + 0.5 * slopes.dhu_y[cD]!;
      const huUpre = hu[cU]! - 0.5 * slopes.dhu_y[cU]!;
      const hvDpre = hv[cD]! + 0.5 * slopes.dhv_y[cD]!;
      const hvUpre = hv[cU]! - 0.5 * slopes.dhv_y[cU]!;

      const bD = bed[cD]!;
      const bU = bed[cU]!;
      const bStar = Math.max(bD, bU);

      const hDpre = Math.max(0, wDpre - bD);
      const hUpre = Math.max(0, wUpre - bU);

      const hStarD = Math.max(0, wDpre - bStar);
      const hStarU = Math.max(0, wUpre - bStar);

      const uDpre = desingularize(hDpre, huDpre, eps);
      const vDpre = desingularize(hDpre, hvDpre, eps);
      const uUpre = desingularize(hUpre, huUpre, eps);
      const vUpre = desingularize(hUpre, hvUpre, eps);
      const huStarD = hStarD * uDpre;
      const hvStarD = hStarD * vDpre;
      const huStarU = hStarU * uUpre;
      const hvStarU = hStarU * vUpre;

      const localMax = centralUpwindFlux(
        hStarD, huStarD, hvStarD,
        hStarU, huStarU, hvStarU,
        1, g, eps, scratchFlux,
      );
      if (localMax > maxSpeed) maxSpeed = localMax;

      dU_h[cD]! -= dtOverDx * scratchFlux.h;
      dU_hu[cD]! -= dtOverDx * scratchFlux.hu;
      dU_hv[cD]! -= dtOverDx * scratchFlux.hv;
      dU_h[cU]! += dtOverDx * scratchFlux.h;
      dU_hu[cU]! += dtOverDx * scratchFlux.hu;
      dU_hv[cU]! += dtOverDx * scratchFlux.hv;

      // Well-balanced y-momentum bed-pressure source (same structure as x)
      const pressDstar = 0.5 * g * hStarD * hStarD;
      const pressUstar = 0.5 * g * hStarU * hStarU;
      dU_hv[cD]! += dtOverDx * pressDstar;
      dU_hv[cU]! -= dtOverDx * pressUstar;
    }
  }

  // 5. Domain-edge faces — handle boundaries via ghost cells.
  //    For each edge cell, compute the flux at the outer face using the
  //    ghost-cell rule appropriate to the cell's boundary type.
  applyEdgeFluxes(state, params, scratchFlux, dU_h, dU_hu, dU_hv);

  // 6. Wall/Solid neighbors on interior faces — already handled by the
  //    BT_SOLID skip above (zero flux).  For Closed cells (interior walls
  //    that should reflect), we treat them like Solid for the flux step:
  //    closed cells have h=0 typically, but we want to reflect momentum.
  //    The standard treatment is: at a Closed cell's face with a wet
  //    neighbor, the ghost cell mirrors the neighbor (same h, reflected
  //    momentum).  We apply this by adding a special closed-face contribution.
  //
  //    For now, Closed at domain edges is handled in applyEdgeFluxes.
  //    Interior Closed cells are rare and can be modelled as Solid.

  return maxSpeed;
}

/**
 * Compute fluxes at the four edges of the domain using ghost-cell rules
 * appropriate to each edge cell's boundary type.  Adds the resulting
 * dU contributions in-place.
 */
function applyEdgeFluxes(
  state: KpState,
  params: KpParams,
  scratchFlux: { h: number; hu: number; hv: number },
  dU_h: Float32Array,
  dU_hu: Float32Array,
  dU_hv: Float32Array,
): void {
  const { width: W, height: H, dx, dt, gravity: g, desingEpsilon: eps, boundary } = params;
  const { h, hu, hv, bed } = state;
  const dtOverDx = dt / dx;
  void dt; // suppress unused warning when manningN = 0

  const processFace = (
    cInside: number,
    hI: number,
    huI: number,
    hvI: number,
    bI: number,
    normalAxis: 0 | 1,
    /** +1 if inside is on the LEFT (or DOWN) side of the face; -1 if on RIGHT (or UP). */
    sign: 1 | -1,
  ): void => {
    if (boundary[cInside]! === BT_SOLID) return;

    const bt = boundary[cInside]!;

    // Decide the ghost cell state (mirrors interior with the boundary rule).
    let hG = hI;
    let huG = huI;
    let hvG = hvI;
    const bG = bI; // bed mirrors across domain edge (flat ghost)

    if (bt === BT_CLOSED) {
      // Reflective: flip the NORMAL momentum component
      if (normalAxis === 0) huG = -huI;
      else hvG = -hvI;
    } else if (bt === BT_OPEN) {
      // Transmissive: ghost = interior (no change)
    } else if (bt === BT_INFLOW) {
      // Pin h to target; keep momentum from interior
      hG = params.boundaryTargetH[cInside]!;
    } else if (bt === BT_SEA) {
      // Pin h to target; momentum zero in ghost
      hG = params.boundaryTargetH[cInside]!;
      huG = 0;
      hvG = 0;
    } else {
      // Interior / Sponge / unknown → reflective default at domain edge
      if (normalAxis === 0) huG = -huI;
      else hvG = -hvI;
    }

    // No slope reconstruction at the edge — use cell-centered values
    const wI = hI + bI;
    const wG = hG + bG;
    const bStar = Math.max(bI, bG);
    const hStarI = Math.max(0, wI - bStar);
    const hStarG = Math.max(0, wG - bStar);

    // Reconstruct momentum at the face, keeping velocity constant under
    // depth rescaling (Audusse momentum reconstruction).
    const hPreI = Math.max(0, wI - bI);
    const hPreG = Math.max(0, wG - bG);
    const huStarI_face = hPreI > 0 ? huI * hStarI / hPreI : 0;
    const hvStarI_face = hPreI > 0 ? hvI * hStarI / hPreI : 0;
    const huStarG_face = hPreG > 0 ? huG * hStarG / hPreG : 0;
    const hvStarG_face = hPreG > 0 ? hvG * hStarG / hPreG : 0;

    let hL: number, huL: number, hvL: number;
    let hR: number, huR: number, hvR: number;
    if (sign === 1) {
      // inside is LEFT (or DOWN); ghost is RIGHT (or UP)
      hL = hStarI; huL = huStarI_face; hvL = hvStarI_face;
      hR = hStarG; huR = huStarG_face; hvR = hvStarG_face;
    } else {
      // inside is RIGHT (or UP); ghost is LEFT (or DOWN)
      hL = hStarG; huL = huStarG_face; hvL = hvStarG_face;
      hR = hStarI; huR = huStarI_face; hvR = hvStarI_face;
    }

    centralUpwindFlux(hL, huL, hvL, hR, huR, hvR, normalAxis, g, eps, scratchFlux);

    // sign = +1: inside is LEFT (face is EAST of inside) → flux is OUTFLOW
    // sign = -1: inside is RIGHT (face is WEST of inside) → flux is INFLOW
    const cellFluxSign = -sign;
    dU_h[cInside]! += cellFluxSign * dtOverDx * scratchFlux.h;
    dU_hu[cInside]! += cellFluxSign * dtOverDx * scratchFlux.hu;
    dU_hv[cInside]! += cellFluxSign * dtOverDx * scratchFlux.hv;

    // Well-balanced pressure source at the edge face
    const pressIstar = 0.5 * g * hStarI * hStarI;
    const cellSourceSign = sign; // matches the face-source convention used inside
    if (normalAxis === 0) {
      dU_hu[cInside]! += cellSourceSign * dtOverDx * pressIstar;
    } else {
      dU_hv[cInside]! += cellSourceSign * dtOverDx * pressIstar;
    }
  };

  // West edge (i = 0): inside is RIGHT of the face
  for (let j = 0; j < H; j++) {
    const c = idx(0, j, W);
    processFace(c, h[c]!, hu[c]!, hv[c]!, bed[c]!, 0, -1);
  }
  // East edge (i = W-1): inside is LEFT of the face
  for (let j = 0; j < H; j++) {
    const c = idx(W - 1, j, W);
    processFace(c, h[c]!, hu[c]!, hv[c]!, bed[c]!, 0, 1);
  }
  // South edge (j = 0): inside is UP of the face
  for (let i = 0; i < W; i++) {
    const c = idx(i, 0, W);
    processFace(c, h[c]!, hu[c]!, hv[c]!, bed[c]!, 1, -1);
  }
  // North edge (j = H-1): inside is DOWN of the face
  for (let i = 0; i < W; i++) {
    const c = idx(i, H - 1, W);
    processFace(c, h[c]!, hu[c]!, hv[c]!, bed[c]!, 1, 1);
  }
}

// ---------------------------------------------------------------------------
// Manning bed friction (semi-implicit, SWASHES §1 eqs. 1–2)
// ---------------------------------------------------------------------------

export function applyManningFriction(state: KpState, params: KpParams): void {
  const { width: W, height: H, dt, gravity: g, manningN: n, desingEpsilon: eps } = params;
  if (n <= 0) return;
  const cells = W * H;
  const n2 = n * n;

  for (let c = 0; c < cells; c++) {
    const h = state.h[c]!;
    if (h <= DRY_THRESHOLD) continue;

    const u = desingularize(h, state.hu[c]!, eps);
    const v = desingularize(h, state.hv[c]!, eps);
    const speed = Math.hypot(u, v);
    if (speed === 0) continue;

    // cf = g · n² / h^(4/3)
    const cf = (g * n2) / Math.pow(h, 4 / 3);
    // Semi-implicit: q^{n+1} = q^n / (1 + dt · cf · |U|)
    const damp = 1 / (1 + dt * cf * speed);
    state.hu[c] = state.hu[c]! * damp;
    state.hv[c] = state.hv[c]! * damp;
  }
}

// ---------------------------------------------------------------------------
// Boundary post-step enforcement (Inflow, Sea, Sponge, Solid pin)
// ---------------------------------------------------------------------------

export function applyBoundaryPostStep(state: KpState, params: KpParams): void {
  const { width: W, height: H, boundary, boundaryTargetH } = params;
  const cells = W * H;
  for (let c = 0; c < cells; c++) {
    const bt = boundary[c]!;
    if (bt === BT_INFLOW) {
      // Pin h to target floor
      state.h[c] = Math.max(state.h[c]!, boundaryTargetH[c]!);
    } else if (bt === BT_SEA) {
      state.h[c] = boundaryTargetH[c]!;
      state.hu[c] = 0;
      state.hv[c] = 0;
    } else if (bt === BT_SOLID) {
      state.h[c] = 0;
      state.hu[c] = 0;
      state.hv[c] = 0;
    } else if (bt === BT_SPONGE) {
      // Damp momentum toward zero (mild) and h toward neighbor mean
      const i = c % W;
      const j = Math.floor(c / W);
      let sumH = 0;
      let count = 0;
      if (i > 0) { sumH += state.h[c - 1]!; count++; }
      if (i < W - 1) { sumH += state.h[c + 1]!; count++; }
      if (j > 0) { sumH += state.h[c - W]!; count++; }
      if (j < H - 1) { sumH += state.h[c + W]!; count++; }
      if (count > 0) {
        const mean = sumH / count;
        state.h[c] = state.h[c]! + 0.1 * (mean - state.h[c]!);
      }
      state.hu[c] = state.hu[c]! * 0.9;
      state.hv[c] = state.hv[c]! * 0.9;
    }
  }
}

// ---------------------------------------------------------------------------
// One full SSP-RK2 (Heun) step
// ---------------------------------------------------------------------------

/**
 * One SSP-RK2 (Heun) step of the KP scheme.  Mutates `state` in place.
 *
 * Returns the maximum wave speed observed during the step, useful for
 * driving adaptive sub-stepping at the caller.
 *
 * Scratch buffers (`scratch`) are allocated once and reused across calls
 * to avoid per-step GC pressure.
 */
export interface KpScratch {
  slopes: KpSlopes;
  /** Backup of (h, hu, hv) at the start of the step for the RK2 average. */
  h0: Float32Array;
  hu0: Float32Array;
  hv0: Float32Array;
  /** dU buffer for one RK stage. */
  dU_h: Float32Array;
  dU_hu: Float32Array;
  dU_hv: Float32Array;
  /** Scratch flux record. */
  flux: { h: number; hu: number; hv: number };
}

export function createKpScratch(width: number, height: number): KpScratch {
  const cells = width * height;
  return {
    slopes: createKpSlopes(width, height),
    h0: new Float32Array(cells),
    hu0: new Float32Array(cells),
    hv0: new Float32Array(cells),
    dU_h: new Float32Array(cells),
    dU_hu: new Float32Array(cells),
    dU_hv: new Float32Array(cells),
    flux: { h: 0, hu: 0, hv: 0 },
  };
}

export function kpStep(
  state: KpState,
  params: KpParams,
  scratch: KpScratch,
): number {
  // Snapshot U^n
  scratch.h0.set(state.h);
  scratch.hu0.set(state.hu);
  scratch.hv0.set(state.hv);

  // Stage 1:  U^(1) = U^n + dt · L(U^n)
  const maxSpeed1 = kpSpatialOperator(
    state, params, scratch.slopes, scratch.flux,
    scratch.dU_h, scratch.dU_hu, scratch.dU_hv,
  );
  for (let i = 0; i < state.h.length; i++) {
    state.h[i] = Math.max(0, state.h[i]! + scratch.dU_h[i]!);
    state.hu[i] = state.hu[i]! + scratch.dU_hu[i]!;
    state.hv[i] = state.hv[i]! + scratch.dU_hv[i]!;
    // If h was clipped to zero, momentum becomes 0 too (no momentum without mass)
    if (state.h[i]! <= DRY_THRESHOLD) {
      state.hu[i] = 0;
      state.hv[i] = 0;
    }
  }
  applyBoundaryPostStep(state, params);

  // Stage 2:  U^(n+1) = ½·U^n + ½·(U^(1) + dt · L(U^(1)))
  const maxSpeed2 = kpSpatialOperator(
    state, params, scratch.slopes, scratch.flux,
    scratch.dU_h, scratch.dU_hu, scratch.dU_hv,
  );
  for (let i = 0; i < state.h.length; i++) {
    const stage1H = state.h[i]!;
    const stage1Hu = state.hu[i]!;
    const stage1Hv = state.hv[i]!;
    state.h[i] = Math.max(0, 0.5 * scratch.h0[i]! + 0.5 * (stage1H + scratch.dU_h[i]!));
    state.hu[i] = 0.5 * scratch.hu0[i]! + 0.5 * (stage1Hu + scratch.dU_hu[i]!);
    state.hv[i] = 0.5 * scratch.hv0[i]! + 0.5 * (stage1Hv + scratch.dU_hv[i]!);
    if (state.h[i]! <= DRY_THRESHOLD) {
      state.hu[i] = 0;
      state.hv[i] = 0;
    }
  }

  // Friction is applied semi-implicitly to the final state
  applyManningFriction(state, params);

  // Boundary post-step enforcement
  applyBoundaryPostStep(state, params);

  return Math.max(maxSpeed1, maxSpeed2);
}

// ---------------------------------------------------------------------------
// Convenience: derive velocity from momentum for inspection / coupling
// ---------------------------------------------------------------------------

/**
 * Fill `outU`, `outV` with desingularized velocities derived from the
 * current momentum field.  Useful for visualization, drag-force kernels,
 * and tests.
 */
export function deriveVelocity(
  state: KpState,
  eps: number,
  outU: Float32Array,
  outV: Float32Array,
): void {
  for (let c = 0; c < state.h.length; c++) {
    outU[c] = desingularize(state.h[c]!, state.hu[c]!, eps);
    outV[c] = desingularize(state.h[c]!, state.hv[c]!, eps);
  }
}

// ---------------------------------------------------------------------------
// CFL dt bound
// ---------------------------------------------------------------------------

/**
 * The maximum stable Δt given the current maxSpeed observation.  CFL is
 * conservative; default target 0.45.
 */
export function kpCflDt(maxSpeed: number, dx: number, cflTarget = 0.45): number {
  if (maxSpeed <= 1e-9) return Infinity;
  return (cflTarget * dx) / maxSpeed;
}

void readGhost; // tree-shake nudge: exported as part of the helper set
