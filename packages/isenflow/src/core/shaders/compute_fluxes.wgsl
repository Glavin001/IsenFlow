// Virtual-pipes flux update (Mei et al. 2007).
//
// Reads water+bed for self+4 neighbors, integrates outflow rates per pipe,
// then scales total outflow so the cell can't drain below empty.
//
// References: lisyarus/webgpu-shallow-water (MIT) — see README for attribution.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bed:      array<f32>;  // 2 floats per cell
@group(0) @binding(2) var<storage, read_write> water:    array<f32>;  // 2 floats per cell
@group(0) @binding(3) var<storage, read_write> fluxLR:   array<f32>;  // 2 floats per cell (L, R)
@group(0) @binding(4) var<storage, read_write> fluxUD:   array<f32>;  // 2 floats per cell (D, U)
@group(0) @binding(5) var<storage, read>       boundary: array<u32>;  // 1 u32 per cell

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }

  let idx = cell_idx(p.x, p.y, w);

  let b_self = bed[idx * 2u + 1u];
  let h_self = water[idx * 2u];
  let eta_self = b_self + h_self;

  let p_L = p + vec2<i32>(-1, 0);
  let p_R = p + vec2<i32>(1, 0);
  let p_D = p + vec2<i32>(0, -1);
  let p_U = p + vec2<i32>(0, 1);

  // For Open boundaries (type 2) at grid edges, out-of-bounds neighbors
  // would get bed_total=1e6 (infinite wall), blocking all outflow.
  // Use transmissive condition: set neighbor eta = eta_self so existing
  // flux momentum carries water out of the domain.
  let bt = boundary[idx];
  let is_open = (bt == 2u);

  let eta_L = select(bed_total_at(&bed, p_L, w, h) + h_at(&water, p_L, w, h), eta_self, is_open && !in_bounds(p_L, w, h));
  let eta_R = select(bed_total_at(&bed, p_R, w, h) + h_at(&water, p_R, w, h), eta_self, is_open && !in_bounds(p_R, w, h));
  let eta_D = select(bed_total_at(&bed, p_D, w, h) + h_at(&water, p_D, w, h), eta_self, is_open && !in_bounds(p_D, w, h));
  let eta_U = select(bed_total_at(&bed, p_U, w, h) + h_at(&water, p_U, w, h), eta_self, is_open && !in_bounds(p_U, w, h));

  let dh_L = eta_self - eta_L;
  let dh_R = eta_self - eta_R;
  let dh_D = eta_self - eta_D;
  let dh_U = eta_self - eta_U;

  // Neighbor depths (needed for depth-dependent pipe area and Manning friction)
  let h_L_nbr = h_at(&water, p_L, w, h);
  let h_R_nbr = h_at(&water, p_R, w, h);
  let h_D_nbr = h_at(&water, p_D, w, h);
  let h_U_nbr = h_at(&water, p_U, w, h);

  // Depth-dependent pipe area: A = h_avg · dx, L = dx → A/L = h_avg.
  // This gives the correct SWE wave speed c = √(g·h) instead of c = √(g·dx).
  // Mei 2007 §3.2.1: for a 2D heightfield the pipe cross-section is the
  // water depth at the face, not the cell area.
  let hpipe_L = max(0.01, 0.5 * (h_self + h_L_nbr));
  let hpipe_R = max(0.01, 0.5 * (h_self + h_R_nbr));
  let hpipe_D = max(0.01, 0.5 * (h_self + h_D_nbr));
  let hpipe_U = max(0.01, 0.5 * (h_self + h_U_nbr));

  // Current outflow rates: lr.x=left, lr.y=right; ud.x=down, ud.y=up
  var lr = vec2<f32>(fluxLR[idx * 2u + 0u], fluxLR[idx * 2u + 1u]);
  var ud = vec2<f32>(fluxUD[idx * 2u + 0u], fluxUD[idx * 2u + 1u]);

  // Mei 2007 eqs. 2-5: accel = g · A/L = g · h_pipe (per face)
  lr.x = max(0.0, lr.x * params.damping + params.dt * params.gravity * hpipe_L * dh_L);
  lr.y = max(0.0, lr.y * params.damping + params.dt * params.gravity * hpipe_R * dh_R);
  ud.x = max(0.0, ud.x * params.damping + params.dt * params.gravity * hpipe_D * dh_D);
  ud.y = max(0.0, ud.y * params.damping + params.dt * params.gravity * hpipe_U * dh_U);

  // Manning bed-friction (SWASHES §1 eqs. 1-2, semi-implicit).
  // cf = g·n²/h^(4/3), attenuate: q' = q / (1 + dt·cf·|u|)
  if (params.manningN > 0.0) {
    let n2 = params.manningN * params.manningN;

    var speed_L  = abs(lr.x) / (params.dx * hpipe_L);
    var cf_L     = params.gravity * n2 / pow(hpipe_L, 4.0/3.0);
    lr.x = lr.x / (1.0 + params.dt * cf_L * speed_L);

    var speed_R  = abs(lr.y) / (params.dx * hpipe_R);
    var cf_R     = params.gravity * n2 / pow(hpipe_R, 4.0/3.0);
    lr.y = lr.y / (1.0 + params.dt * cf_R * speed_R);

    var speed_D  = abs(ud.x) / (params.dx * hpipe_D);
    var cf_D     = params.gravity * n2 / pow(hpipe_D, 4.0/3.0);
    ud.x = ud.x / (1.0 + params.dt * cf_D * speed_D);

    var speed_U  = abs(ud.y) / (params.dx * hpipe_U);
    var cf_U     = params.gravity * n2 / pow(hpipe_U, 4.0/3.0);
    ud.y = ud.y / (1.0 + params.dt * cf_U * speed_U);
  }

  let total_out = (lr.x + lr.y + ud.x + ud.y) * params.dt;
  let volume_available = max(0.0, h_self) * params.dx * params.dx;
  let K = select(min(1.0, volume_available / max(total_out, 1e-9)), 0.0, h_self <= 0.0);
  lr = lr * K;
  ud = ud * K;
  if (bt == 1u) {
    lr = vec2<f32>(0.0, 0.0);
    ud = vec2<f32>(0.0, 0.0);
  }

  fluxLR[idx * 2u + 0u] = lr.x;
  fluxLR[idx * 2u + 1u] = lr.y;
  fluxUD[idx * 2u + 0u] = ud.x;
  fluxUD[idx * 2u + 1u] = ud.y;
}
