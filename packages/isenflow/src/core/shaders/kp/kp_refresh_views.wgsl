// Re-derive the legacy-compatible (h, h_prev) `water` buffer and (u, v)
// `velocity` buffer from the conservative `state` buffer.  Run once per
// SWE step (right after stage 2) so coupling kernels (accumulate_forces,
// apply_displacement) and the renderer see fresh, accurate values.
//
// This is the single point where momentum (hu, hv) becomes velocity (u, v)
// via the Kurganov desingularization — boats, debris, splash particles all
// consume the OUTPUT of this kernel.

@group(0) @binding(0) var<uniform> params:   SimParams;
@group(0) @binding(1) var<storage, read>       state:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> water:    array<f32>;  // 2 per cell (h, h_prev)
@group(0) @binding(3) var<storage, read_write> velocity: array<f32>;  // 2 per cell (u, v)
@group(0) @binding(4) var<storage, read>       state_snap: array<vec4<f32>>;

const KP_VIEW_EPS: f32 = 1.0e-3;
const KP_VIEW_SQRT2: f32 = 1.4142135623730951;

fn view_desingularize(h: f32, q: f32) -> f32 {
  if (h <= 0.0) { return 0.0; }
  let h2 = h * h;
  let h4 = h2 * h2;
  let eps4 = KP_VIEW_EPS * KP_VIEW_EPS * KP_VIEW_EPS * KP_VIEW_EPS;
  return (KP_VIEW_SQRT2 * h * q) / sqrt(h4 + max(h4, eps4));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(params.width);
  let H = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, W, H)) { return; }
  let c = cell_idx(p.x, p.y, W);

  let s = state[c];
  let snap = state_snap[c];

  // Water view: h, h_prev (snap is the state at the START of this step → h_prev)
  water[c * 2u + 0u] = s.x;
  water[c * 2u + 1u] = snap.x;

  // Velocity view: desingularized momentum
  velocity[c * 2u + 0u] = view_desingularize(s.x, s.y);
  velocity[c * 2u + 1u] = view_desingularize(s.x, s.z);
}
