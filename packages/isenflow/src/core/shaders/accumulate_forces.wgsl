// Sum per-cell water→body forces into per-chunk fixed-point atomic accumulators.
//
// Pure computation: no allocations. Buoyancy + hydrostatic + drag from §9.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var bedTex:    texture_storage_2d<rg32float, read>;
@group(0) @binding(2) var waterTex:  texture_storage_2d<rg32float, read>;
@group(0) @binding(3) var velocity:  texture_storage_2d<rg32float, read>;
@group(0) @binding(4) var chunkId:   texture_storage_2d<r32uint,   read>;
@group(0) @binding(5) var chunkVel:  texture_storage_2d<rgba32float, read>;
@group(0) @binding(6) var<storage, read_write> forceAccum: array<atomic<i32>>;     // per chunk × 3
@group(0) @binding(7) var<storage, read_write> torqueAccum: array<atomic<i32>>;    // per chunk × 3
@group(0) @binding(8) var<uniform> chunkCOMs: array<vec4<f32>, 256>;               // x,y,z, _

const RHO:   f32 = 1000.0;
const G:     f32 = 9.81;
const CD:    f32 = 1.0;
const KV:    f32 = 5000.0;
const SCALE: f32 = 10000.0;

fn safe_h(p: vec2<i32>, w: i32, h: i32) -> f32 {
  if (p.x < 0 || p.y < 0 || p.x >= w || p.y >= h) { return 0.0; }
  return textureLoad(waterTex, p).x;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }

  let cid = textureLoad(chunkId, p).x;
  if (cid == 0u) { return; }

  let h_self = textureLoad(waterTex, p).x;
  if (h_self <= 0.0) { return; }

  let vel = textureLoad(velocity, p).xy;
  let cv  = textureLoad(chunkVel, p);

  // Drag (in plane).
  let rx = vel.x - cv.x;
  let rz = vel.y - cv.z;
  let speed = length(vec2<f32>(rx, rz));
  let drag_k = 0.5 * CD * RHO * h_self * speed * params.dx;
  let Fx_drag = drag_k * rx;
  let Fz_drag = drag_k * rz;

  // Hydrostatic horizontal — wall cells against neighboring water.
  let hL = safe_h(p + vec2<i32>(-1, 0), w, h);
  let hR = safe_h(p + vec2<i32>(1, 0),  w, h);
  let hD = safe_h(p + vec2<i32>(0, -1), w, h);
  let hU = safe_h(p + vec2<i32>(0, 1),  w, h);
  let Fx_hyd = 0.5 * RHO * G * (hL * hL - hR * hR) * params.dx;
  let Fz_hyd = 0.5 * RHO * G * (hD * hD - hU * hU) * params.dx;

  // Buoyancy (vertical).
  let bed_total = textureLoad(bedTex, p).y;
  let bed_terrain = textureLoad(bedTex, p).x;
  let submerged_h = min(h_self, max(0.0, bed_total - bed_terrain));
  let F_buoy = RHO * G * params.dx * params.dx * submerged_h;

  // Vertical damping.
  let F_vdamp = -KV * min(1.0, submerged_h / max(0.05, bed_total - bed_terrain)) * cv.y * params.dx * params.dx;

  let Fx = Fx_drag + Fx_hyd;
  let Fy = F_buoy  + F_vdamp;
  let Fz = Fz_drag + Fz_hyd;

  // Torque = r × F where r = cellPos − COM.
  let cellWorld = vec3<f32>(
    (f32(p.x) + 0.5) * params.dx,
    bed_terrain,
    (f32(p.y) + 0.5) * params.dx,
  );
  let com = chunkCOMs[min(cid, 255u)].xyz;
  let r = cellWorld - com;
  let tau = cross(r, vec3<f32>(Fx, Fy, Fz));

  let base = i32(cid) * 3;
  atomicAdd(&forceAccum[base + 0], i32(Fx * SCALE));
  atomicAdd(&forceAccum[base + 1], i32(Fy * SCALE));
  atomicAdd(&forceAccum[base + 2], i32(Fz * SCALE));
  atomicAdd(&torqueAccum[base + 0], i32(tau.x * SCALE));
  atomicAdd(&torqueAccum[base + 1], i32(tau.y * SCALE));
  atomicAdd(&torqueAccum[base + 2], i32(tau.z * SCALE));
}
