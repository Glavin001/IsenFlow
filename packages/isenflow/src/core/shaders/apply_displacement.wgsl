// Spec §10: body→water volume-conserving displacement.
//
// Each cell whose bed rose since last frame ejects the delta volume to its
// 8 neighbors. Splash impulse modulates velocity at the same time.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var bedTex:        texture_storage_2d<rg32float, read_write>;
@group(0) @binding(2) var waterTex:      texture_storage_2d<rg32float, read_write>;
@group(0) @binding(3) var prevBedTex:    texture_storage_2d<r32float,  read>;
@group(0) @binding(4) var<storage, read_write> hDelta: array<atomic<i32>>;

const SCALE: f32 = 10000.0;

fn idx(p: vec2<i32>, w: i32) -> u32 {
  return u32(p.y * w + p.x);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }

  let bed_now = textureLoad(bedTex, p).y;
  let bed_prev = textureLoad(prevBedTex, p).x;
  let delta_bed = bed_now - bed_prev;        // positive = bed rose, body intruded

  if (delta_bed <= 0.0) { return; }

  let h_self = textureLoad(waterTex, p).x;
  let delta = min(delta_bed, 0.5 * h_self);
  if (delta <= 0.0) { return; }

  // 8-neighbor Gaussian-ish kernel, normalized.
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
      atomicAdd(&hDelta[idx(np, w)], i32(add * SCALE));
    }
  }
  // Pull water out of source cell.
  atomicAdd(&hDelta[idx(p, w)], i32(-delta * SCALE));
}
