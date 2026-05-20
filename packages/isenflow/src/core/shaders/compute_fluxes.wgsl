// Virtual-pipes flux update (Mei et al. 2007).
//
// For each cell, compute outflow to each of 4 neighbors based on water-surface
// height difference, then scale all outflows so the cell can't drain past empty.
//
// References: lisyarus/webgpu-shallow-water (MIT) — see README for attribution.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var bedTex:    texture_storage_2d<rg32float, read_write>;
@group(0) @binding(2) var waterTex:  texture_storage_2d<rg32float, read_write>;
@group(0) @binding(3) var fluxLR:    texture_storage_2d<rg32float, read_write>;
@group(0) @binding(4) var fluxUD:    texture_storage_2d<rg32float, read_write>;
@group(0) @binding(5) var boundary:  texture_storage_2d<r32uint,   read>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }

  let b_self = textureLoad(bedTex, p).y;
  let h_self = textureLoad(waterTex, p).x;
  let eta_self = b_self + h_self;

  // Neighbor offsets: L, R, D, U (D=-z, U=+z)
  let p_L = p + vec2<i32>(-1, 0);
  let p_R = p + vec2<i32>(1, 0);
  let p_D = p + vec2<i32>(0, -1);
  let p_U = p + vec2<i32>(0, 1);

  let eta_L = safe_load_bed(bedTex, p_L, w, h) + safe_load_height(waterTex, p_L, w, h);
  let eta_R = safe_load_bed(bedTex, p_R, w, h) + safe_load_height(waterTex, p_R, w, h);
  let eta_D = safe_load_bed(bedTex, p_D, w, h) + safe_load_height(waterTex, p_D, w, h);
  let eta_U = safe_load_bed(bedTex, p_U, w, h) + safe_load_height(waterTex, p_U, w, h);

  // Pressure differential drives flow toward lower surface.
  let dh_L = eta_self - eta_L;
  let dh_R = eta_self - eta_R;
  let dh_D = eta_self - eta_D;
  let dh_U = eta_self - eta_U;

  let accel = params.gravity * params.pipeArea / params.pipeLen;

  // Current outflow rates.
  var f = textureLoad(fluxLR, p);
  var g = textureLoad(fluxUD, p);
  // f.x = outflow to LEFT, f.y = outflow to RIGHT
  // g.x = outflow DOWN (-z), g.y = outflow UP (+z)
  f.x = max(0.0, f.x * params.damping + params.dt * accel * dh_L);
  f.y = max(0.0, f.y * params.damping + params.dt * accel * dh_R);
  g.x = max(0.0, g.x * params.damping + params.dt * accel * dh_D);
  g.y = max(0.0, g.y * params.damping + params.dt * accel * dh_U);

  // Outflow scaling: can't drain below zero.
  let total_out = (f.x + f.y + g.x + g.y) * params.dt;
  let volume_available = max(0.0, h_self) * params.dx * params.dx;
  let K = select(min(1.0, volume_available / max(total_out, 1e-9)), 0.0, h_self <= 0.0);
  f *= K;
  g *= K;

  // Reflective walls: zero out any outflow toward an out-of-bounds or closed boundary.
  let bt = textureLoad(boundary, p).x;
  if (bt == 1u) {
    // closed: do not let interior cells push toward boundary (handled via OOB read);
    // and clear flow when cell itself is wall.
    f = vec2<f32>(0.0);
    g = vec2<f32>(0.0);
  }

  textureStore(fluxLR, p, vec4<f32>(f, 0.0, 0.0));
  textureStore(fluxUD, p, vec4<f32>(g, 0.0, 0.0));
}
