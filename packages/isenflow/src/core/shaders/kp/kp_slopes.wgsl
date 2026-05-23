// KP slope-reconstruction kernel.
//
// Per cell: compute generalized-minmod slopes of (w = h+B, hu, hv) in x and
// y, write into `slopes` buffer (8 floats per cell).  Dry cells (h ≤ ε_dry)
// get zero slopes — preserves positivity trivially.
//
// We slope-limit on (w, hu, hv) — not (h, hu, hv) — because limiting on w
// guarantees the well-balanced C-property (lake at rest stays at rest over
// arbitrary topography).  h at faces is derived as h = w_pre - B_cell.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read>       state:  array<vec4<f32>>;     // (h, hu, hv, _)
@group(0) @binding(2) var<storage, read>       bed:    array<f32>;           // 2 per cell
@group(0) @binding(3) var<storage, read_write> slopes: array<f32>;           // 8 per cell

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(params.width);
  let H = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, W, H)) { return; }
  let c = cell_idx(p.x, p.y, W);

  let s = state[c];
  let hC = s.x;

  let theta = KP_SLOPE_THETA;

  var dw_x: f32 = 0.0;
  var dw_y: f32 = 0.0;
  var dhu_x: f32 = 0.0;
  var dhu_y: f32 = 0.0;
  var dhv_x: f32 = 0.0;
  var dhv_y: f32 = 0.0;

  if (hC > KP_DRY_THRESHOLD) {
    let bC = bed[c * 2u + 1u];
    let wC = hC + bC;

    // X-slopes (need i-1, i, i+1)
    if (p.x > 0 && p.x < W - 1) {
      let cL = cell_idx(p.x - 1, p.y, W);
      let cR = cell_idx(p.x + 1, p.y, W);
      let sL = state[cL];
      let sR = state[cR];
      let wL = sL.x + bed[cL * 2u + 1u];
      let wR = sR.x + bed[cR * 2u + 1u];

      dw_x = kp_minmod3(
        theta * (wC - wL),
        0.5   * (wR - wL),
        theta * (wR - wC),
      );
      dhu_x = kp_minmod3(
        theta * (s.y - sL.y),
        0.5   * (sR.y - sL.y),
        theta * (sR.y - s.y),
      );
      dhv_x = kp_minmod3(
        theta * (s.z - sL.z),
        0.5   * (sR.z - sL.z),
        theta * (sR.z - s.z),
      );
    }

    // Y-slopes (need j-1, j, j+1)
    if (p.y > 0 && p.y < H - 1) {
      let cD = cell_idx(p.x, p.y - 1, W);
      let cU = cell_idx(p.x, p.y + 1, W);
      let sD = state[cD];
      let sU = state[cU];
      let wD = sD.x + bed[cD * 2u + 1u];
      let wU = sU.x + bed[cU * 2u + 1u];

      dw_y = kp_minmod3(
        theta * (wC - wD),
        0.5   * (wU - wD),
        theta * (wU - wC),
      );
      dhu_y = kp_minmod3(
        theta * (s.y - sD.y),
        0.5   * (sU.y - sD.y),
        theta * (sU.y - s.y),
      );
      dhv_y = kp_minmod3(
        theta * (s.z - sD.z),
        0.5   * (sU.z - sD.z),
        theta * (sU.z - s.z),
      );
    }
  }

  let base = c * 8u;
  slopes[base + 0u] = dw_x;
  slopes[base + 1u] = dw_y;
  slopes[base + 2u] = dhu_x;
  slopes[base + 3u] = dhu_y;
  slopes[base + 4u] = dhv_x;
  slopes[base + 5u] = dhv_y;
  // [base + 6u] = max wave speed observation, filled by update kernel
  // [base + 7u] = reserved
}
