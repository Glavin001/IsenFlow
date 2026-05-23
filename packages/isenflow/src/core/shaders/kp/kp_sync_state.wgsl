// Sync the legacy `water` buffer back into the conservative `state` buffer.
//
// `apply_displacement.wgsl` and `fold_h_delta.wgsl` only modify the legacy
// (h, h_prev) `water` buffer (a holdover from the VP solver where that
// was the authoritative state).  In KP the authoritative state is
// `state` (h, hu, hv); we need to push the displacement-induced h change
// back into state.x so the next `kp_slopes` / `kp_update` pass sees it.
//
// Momentum (state.y, state.z) is preserved: the displacement is a mass
// change only.  (A more accurate model would conserve momentum density
// when h drops by scaling hu, hv by the depth ratio — a future
// improvement that requires snapshotting the pre-displacement h.)
//
// Run as the LAST pass of `applyDisplacement()`, after fold_h_delta.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read>       water: array<f32>;        // 2 per cell
@group(0) @binding(2) var<storage, read_write> state: array<vec4<f32>>;  // (h, hu, hv, _)

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(params.width);
  let H = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, W, H)) { return; }
  let c = cell_idx(p.x, p.y, W);

  let h_new = water[c * 2u + 0u];
  var s = state[c];
  // If h dropped to near-zero, momentum must vanish too (otherwise we'd
  // have undefined velocity).  Otherwise preserve (hu, hv) — the body
  // displaced mass, but the momentum stays with the water that's still
  // present.
  if (h_new <= 1.0e-5) {
    s.y = 0.0;
    s.z = 0.0;
  }
  s.x = h_new;
  state[c] = s;
}
