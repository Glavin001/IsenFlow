/**
 * Pure helpers that turn a cell-centered water/bed snapshot into the
 * (W+1) × (H+1) vertex-grid data a heightfield mesh needs.
 *
 * Split out from the demo-side WaterSurface so the math has no GPU/Three
 * dependency and can be unit-tested. The demo glue (BufferAttribute
 * uploads, material wiring) lives in `apps/demos/src/shared/Water.ts`.
 *
 * Why we need this:
 *   The naive approach (sample one cell per vertex, park dry verts at
 *   y=-1000) creates a 1000-metre vertical cliff at every shoreline.
 *   `computeVertexNormals` then averages the cliff-face normal into the
 *   adjacent wet vertex, producing the "spiky" lighting we want to avoid.
 *
 *   Instead we:
 *     - sample at the *corner* between four cells (Marching-Squares style),
 *     - run a small box blur on `h` to take the edge off solver checkerboard,
 *     - keep dry verts at the bed (no cliff) and tag them with wetness=0 so
 *       the demo material fades them out via per-vertex alpha,
 *     - compute analytic normals from the smoothed height field, masking out
 *       dry neighbours so the shoreline never leaks into wet normals.
 */

export interface WaterSurfaceOptions {
  /** Below this depth (m), a cell is considered dry. Default 0.003 (3 mm). */
  readonly wetThreshold?: number;
  /**
   * Smooth ramp width — wetness saturates to 1 once the depth at a vertex
   * exceeds `wetThreshold * fadeBandFactor`. Larger = softer shoreline.
   * Default 6 (≈ 2 cm at dx=3 mm depth threshold).
   */
  readonly fadeBandFactor?: number;
}

const DEFAULTS = {
  wetThreshold: 0.003,
  fadeBandFactor: 6,
} as const;

/**
 * Extract the per-cell water depth channel from the interleaved
 * `[h, h_prev]` array returned by `VirtualPipesSolver.readWater()`.
 * `out` is reused across frames to avoid allocations.
 */
export function extractWaterDepth(
  interleaved: Float32Array,
  out: Float32Array,
): Float32Array {
  const n = out.length;
  for (let i = 0; i < n; i++) out[i] = Math.max(0, interleaved[i * 2] ?? 0);
  return out;
}

/**
 * Extract the per-cell *total* bed channel from the interleaved
 * `[terrain, total]` array returned by `VirtualPipesSolver.readBed()`.
 */
export function extractBedTotal(
  interleaved: Float32Array,
  out: Float32Array,
): Float32Array {
  const n = out.length;
  for (let i = 0; i < n; i++) out[i] = interleaved[i * 2 + 1] ?? 0;
  return out;
}

/**
 * Conservative 3×3 box blur with edge clamping. Returns a fresh array; the
 * input is untouched. `iterations=0` returns a copy. We deliberately
 * include the centre cell in the average so the original peak is preserved
 * to ≈1/9 weight per pass — that's enough to kill solver checkerboard
 * without flattening real waves.
 */
export function smoothScalarField(
  field: Float32Array,
  width: number,
  height: number,
  iterations: number,
  scratch?: Float32Array,
): Float32Array {
  const n = width * height;
  if (field.length !== n) {
    throw new Error(`smoothScalarField: expected length ${n}, got ${field.length}`);
  }
  let cur: Float32Array = new Float32Array(field);
  if (iterations <= 0) return cur;
  let next = scratch && scratch.length === n ? scratch : new Float32Array(n);
  for (let it = 0; it < iterations; it++) {
    for (let j = 0; j < height; j++) {
      const j0 = Math.max(0, j - 1);
      const j1 = Math.min(height - 1, j + 1);
      for (let i = 0; i < width; i++) {
        const i0 = Math.max(0, i - 1);
        const i1 = Math.min(width - 1, i + 1);
        let s = 0;
        let n2 = 0;
        for (let jj = j0; jj <= j1; jj++) {
          const row = jj * width;
          for (let ii = i0; ii <= i1; ii++) {
            s += cur[row + ii]!;
            n2++;
          }
        }
        next[j * width + i] = s / n2;
      }
    }
    const swap = cur;
    cur = next;
    next = swap;
  }
  return cur;
}

export interface VertexSampleResult {
  /** Length (W+1)*(H+1). Vertex world Y at each corner. */
  positionsY: Float32Array;
  /** Length (W+1)*(H+1). 0 = dry (transparent), 1 = fully wet. */
  wetness: Float32Array;
  /** Largest |Y(i,j) - Y(i±1, j±1)| between *wet* neighbours, in metres. */
  maxWetSpike: number;
}

/**
 * Build vertex Y + wetness for a (W+1)×(H+1) plane mesh from cell-centered
 * water depth and total bed elevation.
 *
 * Vertex (i,j) sits at the corner shared by the up-to-four cells
 * (i-1,j-1), (i,j-1), (i-1,j), (i,j). We average:
 *   - bed: over all in-bounds neighbours (smooth terrain hugging when dry).
 *   - eta (= bed+h): over *wet* neighbours only (so a dry-but-low neighbour
 *     can't drag the surface below the actual water level near a wall).
 *
 * `scratch` is an optional reusable output pair; on success the returned
 * arrays alias the supplied buffers.
 */
export function computeWaterVertices(
  waterDepth: Float32Array,
  bedTotal: Float32Array,
  width: number,
  height: number,
  opts: WaterSurfaceOptions = {},
  scratch?: { positionsY: Float32Array; wetness: Float32Array },
): VertexSampleResult {
  const wetThr = opts.wetThreshold ?? DEFAULTS.wetThreshold;
  const fadeBand = Math.max(1, opts.fadeBandFactor ?? DEFAULTS.fadeBandFactor);
  const vw = width + 1;
  const vh = height + 1;
  const vCount = vw * vh;
  if (waterDepth.length !== width * height) {
    throw new Error(`computeWaterVertices: waterDepth len ${waterDepth.length} ≠ ${width * height}`);
  }
  if (bedTotal.length !== width * height) {
    throw new Error(`computeWaterVertices: bedTotal len ${bedTotal.length} ≠ ${width * height}`);
  }

  const Y =
    scratch?.positionsY && scratch.positionsY.length === vCount
      ? scratch.positionsY
      : new Float32Array(vCount);
  const W =
    scratch?.wetness && scratch.wetness.length === vCount
      ? scratch.wetness
      : new Float32Array(vCount);

  const fadeScale = 1 / (wetThr * fadeBand);

  let maxWetSpike = 0;

  for (let j = 0; j < vh; j++) {
    for (let i = 0; i < vw; i++) {
      let etaSum = 0;
      let hMin = Number.POSITIVE_INFINITY;
      let bedSum = 0;
      let wetN = 0;
      let nbrN = 0;
      // Walk the four cells whose top-right corner is this vertex.
      // (di,dj) ∈ {-1,0}².
      for (let dj = -1; dj <= 0; dj++) {
        const cj = j + dj;
        if (cj < 0 || cj >= height) continue;
        const row = cj * width;
        for (let di = -1; di <= 0; di++) {
          const ci = i + di;
          if (ci < 0 || ci >= width) continue;
          const idx = row + ci;
          const bt = bedTotal[idx]!;
          const hc = waterDepth[idx]!;
          bedSum += bt;
          nbrN++;
          if (hc > wetThr) {
            etaSum += bt + hc;
            if (hc < hMin) hMin = hc;
            wetN++;
          }
        }
      }
      const vi = j * vw + i;
      if (nbrN === 0) {
        Y[vi] = 0;
        W[vi] = 0;
        continue;
      }
      if (wetN === 0) {
        // No wet neighbour at all: vertex sits on the terrain. Wetness=0
        // tells the demo material to fade this vertex to alpha=0, so the
        // mesh leaves no halo over dry land. The vertex *position* still
        // hugs the bed so the triangle to the next wet vertex slopes down
        // continuously instead of jumping to y=-1000 like the legacy code.
        Y[vi] = bedSum / nbrN;
        W[vi] = 0;
        continue;
      }
      // At least one wet corner cell. Surface Y is the mean of η over the
      // wet ones, the dry ones contribute only the bed average we keep for
      // the all-dry case.
      const y = etaSum / wetN;
      Y[vi] = y;
      const fillFrac = wetN / nbrN; // 0.25, 0.5, 0.75, 1
      const depthFade = Math.min(1, (hMin === Number.POSITIVE_INFINITY ? 0 : hMin) * fadeScale);
      W[vi] = fillFrac * depthFade;
    }
  }

  // One pass over the wet block to surface the biggest wet-wet jump — handy
  // for unit tests asserting the surface has no remaining cliffs.
  for (let j = 0; j < vh; j++) {
    for (let i = 0; i < vw; i++) {
      const vi = j * vw + i;
      if (W[vi]! <= 0) continue;
      const y = Y[vi]!;
      const checkNbr = (ni: number, nj: number) => {
        if (ni < 0 || nj < 0 || ni >= vw || nj >= vh) return;
        const nvi = nj * vw + ni;
        if (W[nvi]! <= 0) return;
        const d = Math.abs(Y[nvi]! - y);
        if (d > maxWetSpike) maxWetSpike = d;
      };
      checkNbr(i + 1, j);
      checkNbr(i, j + 1);
    }
  }

  return { positionsY: Y, wetness: W, maxWetSpike };
}

/**
 * Analytic per-vertex normal from finite differences of the surface Y,
 * masking out dry neighbours. The shoreline cliff in the *position* array
 * thus never leaks into wet-vertex normals — which is what produced the
 * "spiky" look on the legacy mesh.
 *
 * Output: length vw*vh*3, normalized (nx, ny, nz).
 */
export function computeWaterNormals(
  positionsY: Float32Array,
  wetness: Float32Array,
  vw: number,
  vh: number,
  dx: number,
  out?: Float32Array,
): Float32Array {
  const n = vw * vh;
  if (positionsY.length !== n || wetness.length !== n) {
    throw new Error('computeWaterNormals: positionsY / wetness must be vw*vh');
  }
  const N = out && out.length === n * 3 ? out : new Float32Array(n * 3);
  const wetEnough = (idx: number) => (wetness[idx] ?? 0) > 0.01;
  for (let j = 0; j < vh; j++) {
    for (let i = 0; i < vw; i++) {
      const vi = j * vw + i;
      const o = vi * 3;
      if (!wetEnough(vi)) {
        // Dry: trivial up-normal. The vertex is invisible (alpha=0) so this
        // just keeps the buffer well-defined.
        N[o] = 0;
        N[o + 1] = 1;
        N[o + 2] = 0;
        continue;
      }
      const center = positionsY[vi]!;
      // Replace dry / out-of-bounds neighbours with the centre Y so the
      // gradient at the shoreline is zero on the dry side (instead of
      // following the bed contour).
      const yL = i > 0 && wetEnough(vi - 1) ? positionsY[vi - 1]! : center;
      const yR = i + 1 < vw && wetEnough(vi + 1) ? positionsY[vi + 1]! : center;
      const yD = j > 0 && wetEnough(vi - vw) ? positionsY[vi - vw]! : center;
      const yU = j + 1 < vh && wetEnough(vi + vw) ? positionsY[vi + vw]! : center;
      const ddx = (yR - yL) / (2 * dx);
      const ddz = (yU - yD) / (2 * dx);
      const nx = -ddx;
      const nz = -ddz;
      const len = Math.hypot(nx, 1, nz) || 1;
      N[o] = nx / len;
      N[o + 1] = 1 / len;
      N[o + 2] = nz / len;
    }
  }
  return N;
}

/**
 * In-place exponential smoothing of `prev` toward `target`:
 *   prev[i] += alpha * (target[i] - prev[i])
 *
 * Used to dampen jumps between solver readbacks so the surface motion
 * tracks the simulation without strobing at the readback rate.
 */
export function lerpFieldInPlace(
  prev: Float32Array,
  target: Float32Array,
  alpha: number,
): void {
  if (prev.length !== target.length) {
    throw new Error('lerpFieldInPlace: length mismatch');
  }
  const a = Math.min(1, Math.max(0, alpha));
  for (let i = 0; i < prev.length; i++) {
    prev[i] = prev[i]! + a * (target[i]! - prev[i]!);
  }
}
