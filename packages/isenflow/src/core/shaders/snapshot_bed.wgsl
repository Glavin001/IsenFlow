// Copy current total bed elevation into prevBed for next frame's displacement diff.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read>       bed:     array<f32>;
@group(0) @binding(2) var<storage, read_write> prevBed: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }
  let idx = cell_idx(p.x, p.y, w);
  prevBed[idx] = bed[idx * 2u + 1u];
}
