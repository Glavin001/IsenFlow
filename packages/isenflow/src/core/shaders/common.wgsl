// Shared structs + helpers for IsenFlow SWE solver.
//
// State buffers (storage textures, rg32float / rgba32float):
//   bed       : rg32f    (b_terrain, b_total)         — solid surface (terrain + walls + chunks)
//   water     : rg32f    (h, h_prev)                  — depth + previous frame snapshot
//   fluxLR    : rg32f    (flux_left_to_right, flux_right_to_left)
//   fluxUD    : rg32f    (flux_up, flux_down)         — "up" = +z direction
//   velocity  : rg32f    (u, v)                       — derived (X, Z) cell velocities
//   chunkId   : r32u                                  — owning chunk (0 = free)
//   chunkVel  : rgba32f  (vx, vy, vz, speed)
//   boundary  : r32u                                  — BoundaryType enum
//
// Atomic accumulators:
//   forceAccum:  vec3<i32>  per chunk      (fixed-point N, scale 1e4)
//   torqueAccum: vec3<i32>  per chunk      (fixed-point N·m)
//   hDelta:     i32         per cell       (fixed-point Δh for displacement pass)

struct SimParams {
  width:   u32,
  height:  u32,
  dx:      f32,
  dt:      f32,
  gravity: f32,
  damping: f32,    // pipe damping ∈ (0, 1]
  pipeArea: f32,   // virtual pipe cross-section (≈ dx*dx)
  pipeLen:  f32,   // virtual pipe length (≈ dx)
};

fn in_bounds(p: vec2<i32>, w: i32, h: i32) -> bool {
  return p.x >= 0 && p.y >= 0 && p.x < w && p.y < h;
}

fn safe_load_height(t: texture_storage_2d<rg32float, read_write>, p: vec2<i32>, w: i32, h: i32) -> f32 {
  if (!in_bounds(p, w, h)) { return 0.0; }
  return textureLoad(t, p).x;
}

fn safe_load_bed(t: texture_storage_2d<rg32float, read_write>, p: vec2<i32>, w: i32, h: i32) -> f32 {
  if (!in_bounds(p, w, h)) { return 1e6; }    // out-of-bounds reads as infinite wall
  return textureLoad(t, p).y;
}
