// Per-cell water→body forces summed into per-chunk fixed-point atomic accumulators.
//
// forceAccum layout: [chunk0.fx, chunk0.fy, chunk0.fz, chunk0.tx, chunk0.ty, chunk0.tz, chunk1...]
//
// `chunkCOMs[cid]` packs (com.x, com.y, com.z, halfHeightY) — the .w channel
// is the body's half-extent along Y so we can derive its top/bottom from the
// COM and compute correct buoyancy on partially-submerged floaters.

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read>       bed:        array<f32>;       // 2 per cell
@group(0) @binding(2) var<storage, read>       water:      array<f32>;       // 2 per cell
@group(0) @binding(3) var<storage, read>       velocity:   array<f32>;       // 2 per cell
@group(0) @binding(4) var<storage, read>       chunkId:    array<u32>;       // 1 per cell
@group(0) @binding(5) var<storage, read>       chunkVel:   array<f32>;       // 4 per cell
@group(0) @binding(6) var<storage, read_write> forceAccum: array<atomic<i32>>; // 6 per chunk
@group(0) @binding(7) var<uniform>             chunkCOMs:  array<vec4<f32>, 256>;

const RHO:   f32 = 1000.0;
const G:     f32 = 9.81;
const CD:    f32 = 1.0;
const KV:    f32 = 5000.0;
const SCALE: f32 = 10000.0;

fn safe_h(arr: ptr<storage, array<f32>, read>, p: vec2<i32>, w: i32, h: i32) -> f32 {
  if (!in_bounds(p, w, h)) { return 0.0; }
  return (*arr)[cell_idx(p.x, p.y, w) * 2u];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }
  let idx = cell_idx(p.x, p.y, w);

  let cid = chunkId[idx];
  if (cid == 0u) { return; }

  // h_self is water depth above bed_total (terrain, since dynamic bodies do
  // NOT raise the bed any longer). May be 0 for wall cells — that's OK,
  // hydrostatic horizontal pressure still applies via the neighbor diff.
  let h_self = water[idx * 2u];

  let vel = vec2<f32>(velocity[idx * 2u + 0u], velocity[idx * 2u + 1u]);
  let cv  = vec4<f32>(chunkVel[idx * 4u + 0u], chunkVel[idx * 4u + 1u],
                      chunkVel[idx * 4u + 2u], chunkVel[idx * 4u + 3u]);

  // ----------------- DRAG (only when water present in cell) -----------------
  var Fx_drag: f32 = 0.0;
  var Fz_drag: f32 = 0.0;
  if (h_self > 0.0) {
    let rx = vel.x - cv.x;
    let rz = vel.y - cv.z;
    let speed = length(vec2<f32>(rx, rz));
    let drag_k = 0.5 * CD * RHO * h_self * speed * params.dx;
    Fx_drag = drag_k * rx;
    Fz_drag = drag_k * rz;
  }

  // ----------------- HYDROSTATIC HORIZONTAL (independent of h_self) -----------------
  let hL = safe_h(&water, p + vec2<i32>(-1, 0), w, h);
  let hR = safe_h(&water, p + vec2<i32>(1, 0),  w, h);
  let hD = safe_h(&water, p + vec2<i32>(0, -1), w, h);
  let hU = safe_h(&water, p + vec2<i32>(0, 1),  w, h);
  let Fx_hyd = 0.5 * RHO * G * (hL * hL - hR * hR) * params.dx;
  let Fz_hyd = 0.5 * RHO * G * (hD * hD - hU * hU) * params.dx;

  // ----------------- BUOYANCY (using body COM ± halfY) -----------------
  let bed_terrain = bed[idx * 2u + 0u];
  let com         = chunkCOMs[min(cid, 255u)].xyz;
  let halfY       = chunkCOMs[min(cid, 255u)].w;
  let body_top    = com.y + halfY;
  let body_bot    = com.y - halfY;
  let waterline   = bed_terrain + h_self;
  let top_in_water = min(waterline, body_top);
  let bot_in_water = max(bed_terrain, body_bot);
  let submerged_h = max(0.0, top_in_water - bot_in_water);
  let F_buoy = RHO * G * params.dx * params.dx * submerged_h;

  // ----------------- VERTICAL DAMPING -----------------
  // Damps vertical motion when submerged. submerged_fraction in [0, 1].
  var submerged_frac: f32 = 0.0;
  let body_h_total = max(1e-3, 2.0 * halfY);
  if (submerged_h > 0.0) {
    submerged_frac = clamp(submerged_h / body_h_total, 0.0, 1.0);
  }
  let F_vdamp = -KV * submerged_frac * cv.y * params.dx * params.dx;

  let Fx = Fx_drag + Fx_hyd;
  let Fy = F_buoy  + F_vdamp;
  let Fz = Fz_drag + Fz_hyd;

  let cellWorld = vec3<f32>(
    (f32(p.x) + 0.5) * params.dx,
    bed_terrain,
    (f32(p.y) + 0.5) * params.dx,
  );
  let r = cellWorld - com;
  let tau = cross(r, vec3<f32>(Fx, Fy, Fz));

  let base = i32(cid) * 6;
  atomicAdd(&forceAccum[base + 0], i32(Fx * SCALE));
  atomicAdd(&forceAccum[base + 1], i32(Fy * SCALE));
  atomicAdd(&forceAccum[base + 2], i32(Fz * SCALE));
  atomicAdd(&forceAccum[base + 3], i32(tau.x * SCALE));
  atomicAdd(&forceAccum[base + 4], i32(tau.y * SCALE));
  atomicAdd(&forceAccum[base + 5], i32(tau.z * SCALE));
}
