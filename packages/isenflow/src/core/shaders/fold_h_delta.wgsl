// Fold hDelta (fixed-point) back into water buffer, then zero the deltas.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> water:  array<f32>;
@group(0) @binding(2) var<storage, read_write> hDelta: array<atomic<i32>>;

const SCALE: f32 = 10000.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }
  let idx = cell_idx(p.x, p.y, w);
  let raw = atomicExchange(&hDelta[idx], 0);
  if (raw == 0) { return; }
  let d = f32(raw) / SCALE;
  water[idx * 2u] = max(0.0, water[idx * 2u] + d);
}
