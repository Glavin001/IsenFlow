// Shared helpers for the Kurganov–Petrova SWE solver.
//
// Conservative state per cell: (h, hu, hv) packed as vec4 with .w pad
//   state[idx]    .x = h    (depth, m)
//   state[idx]    .y = hu   (x-momentum density, m²/s)
//   state[idx]    .z = hv   (y-momentum density, m²/s)
//   state[idx]    .w = 0    (reserved)
//
// Bed (mirrors the VP layout for rasterizer compatibility):
//   bed[idx*2u + 0u] = terrain (static)
//   bed[idx*2u + 1u] = total   (terrain + dynamic body footprint)
// KP only reads `total` for its physics.
//
// Slopes per cell (8 floats, stride 8 in a packed buffer):
//   slopes[idx*8u + 0u] = dw_x   (surface elevation w = h+B slope, x)
//   slopes[idx*8u + 1u] = dw_y
//   slopes[idx*8u + 2u] = dhu_x  (x-momentum slope, x direction)
//   slopes[idx*8u + 3u] = dhu_y
//   slopes[idx*8u + 4u] = dhv_x
//   slopes[idx*8u + 5u] = dhv_y
//   slopes[idx*8u + 6u] = max wave speed observed touching this cell (for CFL)
//   slopes[idx*8u + 7u] = reserved
//
// References: see CpuKurganov.ts header for the citations and full algorithm
// notes — this WGSL is a literal port of that CPU oracle.

const KP_SLOPE_THETA: f32 = 1.3;
const KP_DRY_THRESHOLD: f32 = 1e-5;
const KP_SQRT2: f32 = 1.4142135623730951;
const KP_BT_SOLID: u32 = 6u;
const KP_BT_CLOSED: u32 = 1u;
const KP_BT_OPEN: u32 = 2u;
const KP_BT_INFLOW: u32 = 4u;
const KP_BT_SEA: u32 = 5u;
const KP_BT_SPONGE: u32 = 3u;

fn kp_minmod3(a: f32, b: f32, c: f32) -> f32 {
  if (a > 0.0 && b > 0.0 && c > 0.0) {
    return min(a, min(b, c));
  }
  if (a < 0.0 && b < 0.0 && c < 0.0) {
    return max(a, max(b, c));
  }
  return 0.0;
}

/// Kurganov desingularization: u = √2·h·q / √(h⁴ + max(h⁴, ε⁴))
/// Smooth across h → 0, no cliff-like behaviour of `q/max(h, h_min)`.
fn kp_desingularize(h: f32, q: f32, eps: f32) -> f32 {
  if (h <= 0.0) { return 0.0; }
  let h2 = h * h;
  let h4 = h2 * h2;
  let eps4 = eps * eps * eps * eps;
  return (KP_SQRT2 * h * q) / sqrt(h4 + max(h4, eps4));
}

/// Read state from a cell, returning (h, hu, hv, 0) packed as vec4.
/// Out-of-bounds reads return zero.
fn kp_state_at(
  state: ptr<storage, array<vec4<f32>>, read>,
  p: vec2<i32>, w: i32, h: i32,
) -> vec4<f32> {
  if (!in_bounds(p, w, h)) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
  return (*state)[cell_idx(p.x, p.y, w)];
}

/// Read the total bed elevation at a cell (channel 1 of `bed`).
fn kp_bed_at(
  bed: ptr<storage, array<f32>, read>,
  p: vec2<i32>, w: i32, h: i32,
) -> f32 {
  if (!in_bounds(p, w, h)) { return 1.0e6; }  // huge wall for out-of-bounds
  return (*bed)[cell_idx(p.x, p.y, w) * 2u + 1u];
}

struct KpFlux {
  fH:  f32,
  fHu: f32,
  fHv: f32,
  max_speed: f32,
};

/// Central-upwind flux at a single face given the *Audusse-reconstructed*
/// conserved states on both sides.  `axis = 0u` → x-faces (normal velocity
/// is u); `axis = 1u` → y-faces (normal velocity is v).
fn kp_central_upwind(
  hL: f32, huL: f32, hvL: f32,
  hR: f32, huR: f32, hvR: f32,
  axis: u32, g: f32, eps: f32,
) -> KpFlux {
  // Normal and tangential velocities on each side
  var uL: f32; var vL: f32;
  var uR: f32; var vR: f32;
  if (axis == 0u) {
    uL = kp_desingularize(hL, huL, eps);
    vL = kp_desingularize(hL, hvL, eps);
    uR = kp_desingularize(hR, huR, eps);
    vR = kp_desingularize(hR, hvR, eps);
  } else {
    uL = kp_desingularize(hL, hvL, eps);  // "normal" = v
    vL = kp_desingularize(hL, huL, eps);  // "tangent" = u
    uR = kp_desingularize(hR, hvR, eps);
    vR = kp_desingularize(hR, huR, eps);
  }

  let cL = sqrt(g * max(0.0, hL));
  let cR = sqrt(g * max(0.0, hR));

  let aPlus  = max(max(uL + cL, uR + cR), 0.0);
  let aMinus = min(min(uL - cL, uR - cR), 0.0);
  let denom = aPlus - aMinus;
  if (denom < 1e-12) {
    return KpFlux(0.0, 0.0, 0.0, 0.0);
  }

  let pressureL = 0.5 * g * hL * hL;
  let pressureR = 0.5 * g * hR * hR;

  // Physical fluxes F(U) in the NORMAL direction
  var F_L_h: f32; var F_L_hu: f32; var F_L_hv: f32;
  var F_R_h: f32; var F_R_hu: f32; var F_R_hv: f32;
  if (axis == 0u) {
    F_L_h  = huL;
    F_L_hu = huL * uL + pressureL;
    F_L_hv = huL * vL;
    F_R_h  = huR;
    F_R_hu = huR * uR + pressureR;
    F_R_hv = huR * vR;
  } else {
    F_L_h  = hvL;
    F_L_hu = hvL * vL;            // normal=v carries tangential momentum
    F_L_hv = hvL * uL + pressureL; // normal=v carries normal momentum (squared)
    F_R_h  = hvR;
    F_R_hu = hvR * vR;
    F_R_hv = hvR * uR + pressureR;
  }

  let H_h  = (aPlus * F_L_h  - aMinus * F_R_h  + aPlus * aMinus * (hR  - hL )) / denom;
  let H_hu = (aPlus * F_L_hu - aMinus * F_R_hu + aPlus * aMinus * (huR - huL)) / denom;
  let H_hv = (aPlus * F_L_hv - aMinus * F_R_hv + aPlus * aMinus * (hvR - hvL)) / denom;
  let maxA = max(aPlus, -aMinus);

  return KpFlux(H_h, H_hu, H_hv, maxA);
}
