/**
 * Fixed-point encoding for WebGPU atomic-float workaround.
 *
 * WebGPU only supports `atomic<i32>` / `atomic<u32>`. To sum floats across
 * compute threads we quantize: int = round(float * SCALE).
 *
 * SCALE = 1e4 → resolution 1e-4 N; range ±(2^31 / 1e4) ≈ ±2.15e5 N per cell.
 */
export const FIXED_POINT_SCALE = 1e4;

export function encodeFixed(value: number, scale: number = FIXED_POINT_SCALE): number {
  return Math.round(value * scale) | 0;
}

export function decodeFixed(value: number, scale: number = FIXED_POINT_SCALE): number {
  return value / scale;
}

/** WGSL helpers (string) — concatenate into shader source. */
export const FIXED_POINT_WGSL = /* wgsl */ `
const FIXED_POINT_SCALE: f32 = ${FIXED_POINT_SCALE.toFixed(1)};
fn encode_fixed(v: f32) -> i32 { return i32(v * FIXED_POINT_SCALE); }
fn decode_fixed(v: i32) -> f32 { return f32(v) / FIXED_POINT_SCALE; }
`;
