// Copy bedTex.y (current total bed) into prevBed for next frame's displacement diff.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var bedTex:     texture_storage_2d<rg32float, read>;
@group(0) @binding(2) var prevBedTex: texture_storage_2d<r32float,  write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }
  textureStore(prevBedTex, p, vec4<f32>(textureLoad(bedTex, p).y, 0.0, 0.0, 0.0));
}
