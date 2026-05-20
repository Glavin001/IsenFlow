// Reset per-chunk force / torque accumulators (and optionally hDelta).

@group(0) @binding(0) var<storage, read_write> forceAccum: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> torqueAccum: array<atomic<i32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&forceAccum)) { return; }
  atomicStore(&forceAccum[i], 0);
  atomicStore(&torqueAccum[i], 0);
}
