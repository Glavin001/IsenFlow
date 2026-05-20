// Spec §10: body→water volume-conserving displacement.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bed:        array<f32>;       // 2 floats per cell
@group(0) @binding(2) var<storage, read_write> water:      array<f32>;       // 2 floats per cell
@group(0) @binding(3) var<storage, read>       prevBed:    array<f32>;       // 1 float per cell
@group(0) @binding(4) var<storage, read_write> hDelta:     array<atomic<i32>>;

const SCALE: f32 = 10000.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }
  let idx = cell_idx(p.x, p.y, w);

  let bed_now  = bed[idx * 2u + 1u];
  let bed_prev = prevBed[idx];
  let delta_bed = bed_now - bed_prev;
  if (delta_bed <= 0.0) { return; }

  let h_self = water[idx * 2u];
  let delta = min(delta_bed, 0.5 * h_self);
  if (delta <= 0.0) { return; }

  let weights = array<f32, 9>(
    0.0625, 0.125, 0.0625,
    0.125,  0.0,   0.125,
    0.0625, 0.125, 0.0625,
  );
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let np = p + vec2<i32>(dx, dy);
      if (!in_bounds(np, w, h)) { continue; }
      let wi = (dy + 1) * 3 + (dx + 1);
      let wt = weights[wi];
      let add = delta * wt;
      atomicAdd(&hDelta[cell_idx(np.x, np.y, w)], i32(add * SCALE));
    }
  }
  atomicAdd(&hDelta[idx], i32(-delta * SCALE));
}
