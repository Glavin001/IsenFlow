// KP update kernel — computes dU = Δt · L(U) for one RK stage and applies it
// according to (alpha, beta, gamma) blend with the snapshot:
//
//   U_new = alpha · U_snap + beta · U + gamma · dU
//
// SSP-RK2 (Heun):
//   Stage 1: alpha=0,   beta=1,   gamma=1   → U^(1) = U^n + Δt·L(U^n)
//   Stage 2: alpha=0.5, beta=0.5, gamma=0.5 → U^{n+1} = ½U^n + ½(U^(1) + Δt·L(U^(1)))
//
// Per cell, this kernel:
//   1. Reads the cell's own slopes + state, plus 4 neighbors' slopes + state.
//   2. Computes the flux at all 4 faces (each face is computed twice across
//      the grid; this is intentional — keeps the kernel simple, avoids the
//      need for per-face buffers).
//   3. Sums flux contributions + well-balanced bed-pressure source.
//   4. Writes the blended new state back.
//   5. Enforces post-step boundary rules (Inflow/Sea pin, Solid zero,
//      Sponge damping).
//   6. Optionally writes the max wave speed observed at any face back into
//      slopes[c*8+6] for the CFL reduction pass.

struct KpStageParams {
  alpha: f32,        // weight on U_snap
  beta:  f32,        // weight on U  (current state before applying dU)
  gamma: f32,        // weight on dU
  applyManning: f32, // 1.0 only on last stage
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<uniform> stage:  KpStageParams;
@group(0) @binding(2) var<storage, read_write> state:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       state_snap: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       bed:      array<f32>;
@group(0) @binding(5) var<storage, read_write> slopes:   array<f32>;
@group(0) @binding(6) var<storage, read>       boundary: array<u32>;
@group(0) @binding(7) var<storage, read>       boundaryTargetH: array<f32>;

fn read_slope_dw_x(c: u32) -> f32 { return slopes[c * 8u + 0u]; }
fn read_slope_dw_y(c: u32) -> f32 { return slopes[c * 8u + 1u]; }
fn read_slope_dhu_x(c: u32) -> f32 { return slopes[c * 8u + 2u]; }
fn read_slope_dhu_y(c: u32) -> f32 { return slopes[c * 8u + 3u]; }
fn read_slope_dhv_x(c: u32) -> f32 { return slopes[c * 8u + 4u]; }
fn read_slope_dhv_y(c: u32) -> f32 { return slopes[c * 8u + 5u]; }

struct ReconFace {
  hStar:  f32,
  huStar: f32,
  hvStar: f32,
  /// Pre-Audusse depth on THIS cell's side (used for the well-balanced
  /// bed-pressure source correction: (g/2)·(h*² - h_pre²)).
  hPre:   f32,
};

/// Reconstruct the side-of-face state given the cell's center state +
/// slopes, the offset in slope-units (±0.5), the cell's bed, and the
/// face-effective bed b★ = max(bed_L, bed_R).
fn reconstruct(
  s: vec4<f32>, b: f32, b_star: f32,
  dw: f32, dhu: f32, dhv: f32,
  sign: f32, eps: f32,
) -> ReconFace {
  let w_pre = s.x + b + sign * 0.5 * dw;
  let hu_pre = s.y + sign * 0.5 * dhu;
  let hv_pre = s.z + sign * 0.5 * dhv;
  let h_pre = max(0.0, w_pre - b);              // depth on this cell's edge
  let h_star = max(0.0, w_pre - b_star);        // Audusse-clipped depth at face

  let u_pre = kp_desingularize(h_pre, hu_pre, eps);
  let v_pre = kp_desingularize(h_pre, hv_pre, eps);

  return ReconFace(h_star, h_star * u_pre, h_star * v_pre, h_pre);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(params.width);
  let H = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, W, H)) { return; }
  let c = cell_idx(p.x, p.y, W);

  let bt = boundary[c];
  let s = state[c];

  // SOLID: pin to zero, skip all flux
  if (bt == KP_BT_SOLID) {
    state[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    slopes[c * 8u + 6u] = 0.0;
    return;
  }

  let b_self = bed[c * 2u + 1u];
  let eps = params.dt;       // (using params.dt slot is a placeholder; we
                             //  set the actual eps via params field; see
                             //  below — overridden by per-step uniform)
  // The "real" eps: hard-coded to 1e-3 m for now (KP standard).  When we
  // add a desingEps field to SimParams in a follow-up, swap this out.
  let desEps: f32 = 1.0e-3;

  let dtOverDx = params.dt / params.dx;
  let g = params.gravity;

  // Accumulate dU and max wave speed
  var dU = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var maxSpeed: f32 = 0.0;

  // Self slopes
  let dw_x_s = read_slope_dw_x(c);
  let dw_y_s = read_slope_dw_y(c);
  let dhu_x_s = read_slope_dhu_x(c);
  let dhu_y_s = read_slope_dhu_y(c);
  let dhv_x_s = read_slope_dhv_x(c);
  let dhv_y_s = read_slope_dhv_y(c);

  // ---------------- EAST face (between self and (i+1, j)) ----------------
  // Treat Solid neighbors AND out-of-domain as REFLECTIVE walls (mirror
  // ghost): h_ghost = h_self, hu_ghost = -hu_self (normal flipped), hv
  // preserved.  This keeps the wall-reaction pressure in the flux.
  {
    let pN = p + vec2<i32>(1, 0);
    let inside = in_bounds(pN, W, H);
    let cN = select(c, cell_idx(pN.x, pN.y, W), inside);
    let btN = select(KP_BT_OPEN, boundary[cN], inside);
    let neighborIsSolid = btN == KP_BT_SOLID;

    var sN: vec4<f32>;
    var bN: f32;
    var dw_x_N: f32 = 0.0;
    var dhu_x_N: f32 = 0.0;
    var dhv_x_N: f32 = 0.0;
    if (inside && !neighborIsSolid) {
      sN = state[cN];
      bN = bed[cN * 2u + 1u];
      dw_x_N  = read_slope_dw_x(cN);
      dhu_x_N = read_slope_dhu_x(cN);
      dhv_x_N = read_slope_dhv_x(cN);
    } else if (neighborIsSolid) {
      // Reflective mirror of self (flip x-momentum)
      sN = vec4<f32>(s.x, -s.y, s.z, 0.0);
      bN = b_self;
    } else {
      // Domain edge — ghost based on self boundary type
      sN = s;
      bN = b_self;
      if (bt == KP_BT_CLOSED) { sN.y = -s.y; }
      else if (bt == KP_BT_INFLOW) { sN.x = boundaryTargetH[c]; }
      else if (bt == KP_BT_SEA) { sN.x = boundaryTargetH[c]; sN.y = 0.0; sN.z = 0.0; }
      else if (bt != KP_BT_OPEN) { sN.y = -s.y; }
    }
    let bStar = max(b_self, bN);

    let L = reconstruct(s,  b_self, bStar,
      dw_x_s, dhu_x_s, dhv_x_s, 1.0, desEps);
    let R = reconstruct(sN, bN, bStar,
      dw_x_N, dhu_x_N, dhv_x_N, -1.0, desEps);

    let f = kp_central_upwind(
      L.hStar, L.huStar, L.hvStar,
      R.hStar, R.huStar, R.hvStar,
      0u, g, desEps,
    );
    if (f.max_speed > maxSpeed) { maxSpeed = f.max_speed; }
    dU.x -= dtOverDx * f.fH;
    dU.y -= dtOverDx * f.fHu;
    dU.z -= dtOverDx * f.fHv;
    // L-side Audusse correction: dU_hu += dtOverDx · (g/2) · (h*_L² - h_L²)
    dU.y += dtOverDx * 0.5 * g * (L.hStar * L.hStar - L.hPre * L.hPre);
  }

  // ---------------- WEST face (between self and (i-1, j)) ----------------
  {
    let pN = p + vec2<i32>(-1, 0);
    let inside = in_bounds(pN, W, H);
    let cN = select(c, cell_idx(pN.x, pN.y, W), inside);
    let btN = select(KP_BT_OPEN, boundary[cN], inside);
    let neighborIsSolid = btN == KP_BT_SOLID;

    var sN: vec4<f32>;
    var bN: f32;
    var dw_x_N: f32 = 0.0;
    var dhu_x_N: f32 = 0.0;
    var dhv_x_N: f32 = 0.0;
    if (inside && !neighborIsSolid) {
      sN = state[cN];
      bN = bed[cN * 2u + 1u];
      dw_x_N  = read_slope_dw_x(cN);
      dhu_x_N = read_slope_dhu_x(cN);
      dhv_x_N = read_slope_dhv_x(cN);
    } else if (neighborIsSolid) {
      sN = vec4<f32>(s.x, -s.y, s.z, 0.0);
      bN = b_self;
    } else {
      sN = s;
      bN = b_self;
      if (bt == KP_BT_CLOSED) { sN.y = -s.y; }
      else if (bt == KP_BT_INFLOW) { sN.x = boundaryTargetH[c]; }
      else if (bt == KP_BT_SEA) { sN.x = boundaryTargetH[c]; sN.y = 0.0; sN.z = 0.0; }
      else if (bt != KP_BT_OPEN) { sN.y = -s.y; }
    }
    let bStar = max(b_self, bN);

    let L = reconstruct(sN, bN, bStar,
      dw_x_N, dhu_x_N, dhv_x_N, 1.0, desEps);
    let R = reconstruct(s, b_self, bStar,
      dw_x_s, dhu_x_s, dhv_x_s, -1.0, desEps);

    let f = kp_central_upwind(
      L.hStar, L.huStar, L.hvStar,
      R.hStar, R.huStar, R.hvStar,
      0u, g, desEps,
    );
    if (f.max_speed > maxSpeed) { maxSpeed = f.max_speed; }
    dU.x += dtOverDx * f.fH;
    dU.y += dtOverDx * f.fHu;
    dU.z += dtOverDx * f.fHv;
    // R-side Audusse correction (opposite sign from L-side):
    //   dU_hu += dtOverDx · (g/2) · (h_pre_R² - h*_R²)
    dU.y += dtOverDx * 0.5 * g * (R.hPre * R.hPre - R.hStar * R.hStar);
  }

  // ---------------- NORTH face (between self and (i, j+1)) ----------------
  {
    let pN = p + vec2<i32>(0, 1);
    let inside = in_bounds(pN, W, H);
    let cN = select(c, cell_idx(pN.x, pN.y, W), inside);
    let btN = select(KP_BT_OPEN, boundary[cN], inside);
    let neighborIsSolid = btN == KP_BT_SOLID;

    var sN: vec4<f32>;
    var bN: f32;
    var dw_y_N: f32 = 0.0;
    var dhu_y_N: f32 = 0.0;
    var dhv_y_N: f32 = 0.0;
    if (inside && !neighborIsSolid) {
      sN = state[cN];
      bN = bed[cN * 2u + 1u];
      dw_y_N  = read_slope_dw_y(cN);
      dhu_y_N = read_slope_dhu_y(cN);
      dhv_y_N = read_slope_dhv_y(cN);
    } else if (neighborIsSolid) {
      sN = vec4<f32>(s.x, s.y, -s.z, 0.0);
      bN = b_self;
    } else {
      sN = s;
      bN = b_self;
      if (bt == KP_BT_CLOSED) { sN.z = -s.z; }
      else if (bt == KP_BT_INFLOW) { sN.x = boundaryTargetH[c]; }
      else if (bt == KP_BT_SEA) { sN.x = boundaryTargetH[c]; sN.y = 0.0; sN.z = 0.0; }
      else if (bt != KP_BT_OPEN) { sN.z = -s.z; }
    }
    let bStar = max(b_self, bN);

    let L = reconstruct(s, b_self, bStar,
      dw_y_s, dhu_y_s, dhv_y_s, 1.0, desEps);
    let R = reconstruct(sN, bN, bStar,
      dw_y_N, dhu_y_N, dhv_y_N, -1.0, desEps);

    let f = kp_central_upwind(
      L.hStar, L.huStar, L.hvStar,
      R.hStar, R.huStar, R.hvStar,
      1u, g, desEps,
    );
    if (f.max_speed > maxSpeed) { maxSpeed = f.max_speed; }
    dU.x -= dtOverDx * f.fH;
    dU.y -= dtOverDx * f.fHu;
    dU.z -= dtOverDx * f.fHv;
    dU.z += dtOverDx * 0.5 * g * (L.hStar * L.hStar - L.hPre * L.hPre);
  }

  // ---------------- SOUTH face (between self and (i, j-1)) ----------------
  {
    let pN = p + vec2<i32>(0, -1);
    let inside = in_bounds(pN, W, H);
    let cN = select(c, cell_idx(pN.x, pN.y, W), inside);
    let btN = select(KP_BT_OPEN, boundary[cN], inside);
    let neighborIsSolid = btN == KP_BT_SOLID;

    var sN: vec4<f32>;
    var bN: f32;
    var dw_y_N: f32 = 0.0;
    var dhu_y_N: f32 = 0.0;
    var dhv_y_N: f32 = 0.0;
    if (inside && !neighborIsSolid) {
      sN = state[cN];
      bN = bed[cN * 2u + 1u];
      dw_y_N  = read_slope_dw_y(cN);
      dhu_y_N = read_slope_dhu_y(cN);
      dhv_y_N = read_slope_dhv_y(cN);
    } else if (neighborIsSolid) {
      sN = vec4<f32>(s.x, s.y, -s.z, 0.0);
      bN = b_self;
    } else {
      sN = s;
      bN = b_self;
      if (bt == KP_BT_CLOSED) { sN.z = -s.z; }
      else if (bt == KP_BT_INFLOW) { sN.x = boundaryTargetH[c]; }
      else if (bt == KP_BT_SEA) { sN.x = boundaryTargetH[c]; sN.y = 0.0; sN.z = 0.0; }
      else if (bt != KP_BT_OPEN) { sN.z = -s.z; }
    }
    let bStar = max(b_self, bN);

    let L = reconstruct(sN, bN, bStar,
      dw_y_N, dhu_y_N, dhv_y_N, 1.0, desEps);
    let R = reconstruct(s, b_self, bStar,
      dw_y_s, dhu_y_s, dhv_y_s, -1.0, desEps);

    let f = kp_central_upwind(
      L.hStar, L.huStar, L.hvStar,
      R.hStar, R.huStar, R.hvStar,
      1u, g, desEps,
    );
    if (f.max_speed > maxSpeed) { maxSpeed = f.max_speed; }
    dU.x += dtOverDx * f.fH;
    dU.y += dtOverDx * f.fHu;
    dU.z += dtOverDx * f.fHv;
    dU.z += dtOverDx * 0.5 * g * (R.hPre * R.hPre - R.hStar * R.hStar);
  }

  // ---------------- RK blend + apply ----------------
  let snap = state_snap[c];
  var newState = stage.alpha * snap + stage.beta * s + stage.gamma * dU;
  newState.x = max(0.0, newState.x);
  if (newState.x <= KP_DRY_THRESHOLD) {
    newState.y = 0.0;
    newState.z = 0.0;
  }

  // ---------------- Boundary post-step ----------------
  if (bt == KP_BT_INFLOW) {
    newState.x = max(newState.x, boundaryTargetH[c]);
  } else if (bt == KP_BT_SEA) {
    newState.x = boundaryTargetH[c];
    newState.y = 0.0;
    newState.z = 0.0;
  } else if (bt == KP_BT_SPONGE) {
    // Damp momentum, blend h toward neighbor mean
    var sumH: f32 = 0.0;
    var count: f32 = 0.0;
    if (p.x > 0)     { sumH += state[cell_idx(p.x - 1, p.y, W)].x; count += 1.0; }
    if (p.x < W - 1) { sumH += state[cell_idx(p.x + 1, p.y, W)].x; count += 1.0; }
    if (p.y > 0)     { sumH += state[cell_idx(p.x, p.y - 1, W)].x; count += 1.0; }
    if (p.y < H - 1) { sumH += state[cell_idx(p.x, p.y + 1, W)].x; count += 1.0; }
    if (count > 0.0) {
      newState.x = newState.x + 0.1 * (sumH / count - newState.x);
    }
    newState.y = newState.y * 0.9;
    newState.z = newState.z * 0.9;
  }

  // ---------------- Manning friction (on final stage only) ----------------
  if (stage.applyManning > 0.5 && params.manningN > 0.0 && newState.x > KP_DRY_THRESHOLD) {
    let u = kp_desingularize(newState.x, newState.y, desEps);
    let v = kp_desingularize(newState.x, newState.z, desEps);
    let speed = sqrt(u * u + v * v);
    if (speed > 0.0) {
      let n2 = params.manningN * params.manningN;
      let cf = g * n2 / pow(newState.x, 4.0 / 3.0);
      let damp = 1.0 / (1.0 + params.dt * cf * speed);
      newState.y = newState.y * damp;
      newState.z = newState.z * damp;
    }
  }

  state[c] = newState;
  slopes[c * 8u + 6u] = maxSpeed;
}
