// Shared structs + helpers for IsenFlow SWE solver.
//
// All cell-indexed state lives in storage buffers (not storage textures) so
// we don't depend on the WebGPU `chromium-experimental-read-write-storage-texture`
// feature and aren't gated by per-stage storage-texture limits.
//
// Cell index: idx = j*width + i
//
// 2-component cells (water, flux, velocity, bed) store .x at idx*2, .y at idx*2+1.
// 4-component cells (chunkVel) store .x..w at idx*4..idx*4+3.

struct SimParams {
  width:    u32,
  height:   u32,
  dx:       f32,
  dt:       f32,
  gravity:  f32,
  damping:  f32,
  pipeArea: f32,
  pipeLen:  f32,
  manningN: f32,
  originX:  f32,
  originZ:  f32,
  cpuBuoyancyMode: f32,
};

fn cell_idx(i: i32, j: i32, w: i32) -> u32 { return u32(j * w + i); }

fn in_bounds(p: vec2<i32>, w: i32, h: i32) -> bool {
  return p.x >= 0 && p.y >= 0 && p.x < w && p.y < h;
}

// Read helpers that clamp to 0 (or "infinite wall" for bed) when out of bounds.
fn h_at(arr: ptr<storage, array<f32>, read_write>, p: vec2<i32>, w: i32, h: i32) -> f32 {
  if (!in_bounds(p, w, h)) { return 0.0; }
  return (*arr)[cell_idx(p.x, p.y, w) * 2u];
}
fn bed_total_at(arr: ptr<storage, array<f32>, read_write>, p: vec2<i32>, w: i32, h: i32) -> f32 {
  if (!in_bounds(p, w, h)) { return 1.0e6; }
  return (*arr)[cell_idx(p.x, p.y, w) * 2u + 1u];
}
