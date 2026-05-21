/**
 * Pragmatic water-surface renderer: a plane geometry whose vertices are
 * updated from a CPU-mirror of the water depth texture every few frames.
 *
 * For production you'd port this to a vertex shader that samples the GPU
 * texture directly (see WaterMaterial in the library). Until then the
 * displacement runs on the CPU using `waterSurfaceMath`, which:
 *   - corner-samples h and bed at each vertex (no per-cell aliasing),
 *   - runs a 3x3 box blur to take the edge off solver checkerboard,
 *   - keeps dry vertices hugging the terrain so the shoreline has no cliff,
 *   - tags each vertex with a "wetness" in [0,1] that drives per-vertex
 *     alpha — dry verts fade to fully transparent, wet verts stay opaque.
 *
 * Why all of this matters: the legacy version did
 *   y = h > ε ? bed + h : -1000;
 * which left a 1000-metre cliff between any adjacent wet/dry vertices.
 * `computeVertexNormals` then averaged the cliff-face normal into every
 * shoreline-adjacent wet vertex — that's where the "spiky", non-water-like
 * highlights came from.
 */
import * as THREE from 'three';
import type { VirtualPipesSolver } from 'isenflow';
import {
  computeWaterVertices,
  computeWaterNormals,
  smoothScalarField,
  extractWaterDepth,
  extractBedTotal,
  lerpFieldInPlace,
} from 'isenflow';

const READBACK_ERROR_THRESHOLD = 5;

export interface WaterSurfaceOpts {
  /** Frames between async GPU readbacks. Default 3. */
  readbackInterval?: number;
  /** 3x3 box-blur passes on h before sampling. Default 1. */
  smoothPasses?: number;
  /** Per-frame exponential smoothing toward the latest sampled Y. Default 0.35. */
  temporalLerp?: number;
  /** Depth (m) under which a cell is considered dry. Default 0.003. */
  wetThreshold?: number;
}

export class WaterSurface {
  readonly mesh: THREE.Mesh;
  readonly geom: THREE.PlaneGeometry;
  private positions: Float32BufferAttribute;
  private normals: Float32BufferAttribute;
  private colorAttr: THREE.BufferAttribute;
  private readonly opts: Required<WaterSurfaceOpts>;

  private framesSinceRead = 0;
  private reading = false;
  private consecutiveErrors = 0;
  private lastErrorMessage = '';
  /** True once the first successful readback has populated vertex Y values. */
  hasFreshData = false;
  /** Raw GPU readback from last successful update (interleaved [h, h_prev]). */
  lastWaterData: Float32Array | null = null;
  /** Raw GPU readback from last successful update (interleaved [terrain, total]). */
  lastBedData: Float32Array | null = null;

  // Scratch buffers — reused across frames to avoid alloc churn.
  private readonly depthCells: Float32Array;
  private readonly bedCells: Float32Array;
  private readonly smoothScratch: Float32Array;
  private readonly targetY: Float32Array;
  private readonly currentY: Float32Array;
  private readonly targetWetness: Float32Array;
  private readonly currentWetness: Float32Array;
  private readonly normalScratch: Float32Array;
  private targetReady = false;

  constructor(private readonly solver: VirtualPipesSolver, opts: WaterSurfaceOpts = {}) {
    this.opts = {
      readbackInterval: opts.readbackInterval ?? 3,
      smoothPasses: opts.smoothPasses ?? 1,
      temporalLerp: opts.temporalLerp ?? 0.35,
      wetThreshold: opts.wetThreshold ?? 0.003,
    };
    const g = solver.grid;
    this.geom = new THREE.PlaneGeometry(g.width * g.dx, g.height * g.dx, g.width, g.height);
    this.geom.rotateX(-Math.PI / 2);
    this.positions = this.geom.attributes.position as Float32BufferAttribute;
    this.normals = this.geom.attributes.normal as Float32BufferAttribute;

    const vw = g.width + 1;
    const vh = g.height + 1;
    const vCount = vw * vh;

    // Per-vertex RGBA. RGB stays (1,1,1) so the material base colour shows
    // through unchanged; alpha tracks wetness. Three.js MeshStandardMaterial
    // with vertexColors=true and itemSize=4 honours vertex alpha in the
    // final pixel alpha. If a renderer ignores it, dry verts at least sit
    // at bed level (no cliff), so the worst case is a faint blue tint over
    // dry land — still infinitely better than 1000 m spikes.
    const colors = new Float32Array(vCount * 4);
    for (let i = 0; i < vCount; i++) {
      colors[i * 4 + 0] = 1;
      colors[i * 4 + 1] = 1;
      colors[i * 4 + 2] = 1;
      colors[i * 4 + 3] = 0; // start fully dry
    }
    this.colorAttr = new THREE.BufferAttribute(colors, 4);
    this.geom.setAttribute('color', this.colorAttr);

    this.depthCells = new Float32Array(g.cells);
    this.bedCells = new Float32Array(g.cells);
    this.smoothScratch = new Float32Array(g.cells);
    this.targetY = new Float32Array(vCount);
    this.currentY = new Float32Array(vCount);
    this.targetWetness = new Float32Array(vCount);
    this.currentWetness = new Float32Array(vCount);
    this.normalScratch = new Float32Array(vCount * 3);

    // Park everything below the world until the first readback lands.
    this.currentY.fill(-1000);
    this.targetY.fill(-1000);
    for (let i = 0; i < vCount; i++) this.positions.setY(i, -1000);
    this.positions.needsUpdate = true;

    const mat = new THREE.MeshStandardMaterial({
      color: 0x3870c8,
      transparent: true,
      // Slightly higher base opacity than legacy (0.85). Per-vertex alpha
      // takes the surface fully transparent on dry land, so the global
      // opacity controls the wet body of water only.
      opacity: 0.92,
      roughness: 0.18,
      metalness: 0.08,
      side: THREE.DoubleSide,
      vertexColors: true,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(this.geom, mat);
    this.mesh.name = 'water';
    this.mesh.position.set(
      g.origin[0] + (g.width * g.dx) / 2,
      0,
      g.origin[1] + (g.height * g.dx) / 2,
    );
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1; // render after solid scene
  }

  /**
   * Drive the every-N-frames GPU readback. Demos call this once per frame;
   * heavy work (smoothing, corner sampling, normal compute, geometry
   * upload) happens once per successful readback, *not* every frame —
   * doing it per-frame is enough to push real-GPU demos over their 60 FPS
   * perf budget on a 384² grid.
   */
  update(): void {
    if (this.reading) return;
    this.framesSinceRead++;
    if (this.framesSinceRead < this.opts.readbackInterval) return;
    this.framesSinceRead = 0;
    this.reading = true;
    this.kickReadback();
  }

  private kickReadback(): void {
    Promise.all([this.solver.readWater(), this.solver.readBed()])
      .then(([w, b]) => {
        this.lastWaterData = w;
        this.lastBedData = b;
        const g = this.solver.grid;

        // 1) extract single-channel h, bedTotal arrays from interleaved data.
        extractWaterDepth(w, this.depthCells);
        extractBedTotal(b, this.bedCells);

        // 2) smooth h to remove solver-side high-frequency content. This
        //    is purely for display — the solver continues to integrate the
        //    raw signal.
        const hSmoothed =
          this.opts.smoothPasses > 0
            ? smoothScalarField(this.depthCells, g.width, g.height, this.opts.smoothPasses, this.smoothScratch)
            : this.depthCells;

        // 3) corner-sample to (W+1)*(H+1) vertex arrays.
        computeWaterVertices(
          hSmoothed,
          this.bedCells,
          g.width,
          g.height,
          { wetThreshold: this.opts.wetThreshold },
          { positionsY: this.targetY, wetness: this.targetWetness },
        );

        // 4) optional one-shot temporal blend with the previous frame's
        //    sampled state. We do this *here* (not per-frame) so we still
        //    smooth between readbacks but only pay the cost once per
        //    readback. With temporalLerp=0 the blend is a no-op.
        if (!this.targetReady) {
          this.currentY.set(this.targetY);
          this.currentWetness.set(this.targetWetness);
        } else if (this.opts.temporalLerp < 1) {
          lerpFieldInPlace(this.currentY, this.targetY, this.opts.temporalLerp);
          lerpFieldInPlace(this.currentWetness, this.targetWetness, this.opts.temporalLerp);
        } else {
          this.currentY.set(this.targetY);
          this.currentWetness.set(this.targetWetness);
        }

        // 5) write Y + alpha into the BufferAttributes, recompute normals.
        this.uploadGeometry();
        this.targetReady = true;
        this.hasFreshData = true;
        this.consecutiveErrors = 0;
      })
      .catch((err) => {
        this.consecutiveErrors++;
        this.lastErrorMessage = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
        // eslint-disable-next-line no-console
        console.warn(
          `[WaterSurface] readback failed (${this.consecutiveErrors}× in a row): ${this.lastErrorMessage}`,
        );
        if (this.consecutiveErrors >= READBACK_ERROR_THRESHOLD) {
          this.showFatalBanner();
        }
      })
      .finally(() => {
        this.reading = false;
      });
  }

  private uploadGeometry(): void {
    const g = this.solver.grid;
    const vw = g.width + 1;
    const vh = g.height + 1;
    const vCount = vw * vh;

    const posArr = this.positions.array as Float32Array;
    const colArr = this.colorAttr.array as Float32Array;
    // PlaneGeometry+rotateX(-π/2): vertex layout is [x, y, z] per vertex,
    // and Y is index*3 + 1.
    for (let vi = 0; vi < vCount; vi++) {
      posArr[vi * 3 + 1] = this.currentY[vi]!;
      colArr[vi * 4 + 3] = this.currentWetness[vi]!;
    }
    this.positions.needsUpdate = true;
    this.colorAttr.needsUpdate = true;

    // Analytic normals from the smoothed Y field, masking out dry verts so
    // shoreline transitions never leak into wet-vertex normals.
    computeWaterNormals(this.currentY, this.currentWetness, vw, vh, g.dx, this.normalScratch);
    const nArr = this.normals.array as Float32Array;
    nArr.set(this.normalScratch);
    this.normals.needsUpdate = true;
  }

  /** Last surfaced error message — useful for tests. */
  get readbackError(): { count: number; message: string } {
    return { count: this.consecutiveErrors, message: this.lastErrorMessage };
  }

  /**
   * Largest |ΔY| between adjacent wet vertices, in metres. Tests can read
   * this to assert the rendered surface has no cliffs.
   */
  get maxWetSpike(): number {
    const g = this.solver.grid;
    const vw = g.width + 1, vh = g.height + 1;
    let m = 0;
    for (let j = 0; j < vh; j++) {
      for (let i = 0; i < vw; i++) {
        const vi = j * vw + i;
        if (this.currentWetness[vi]! < 0.05) continue;
        const y = this.currentY[vi]!;
        if (i + 1 < vw) {
          const nv = vi + 1;
          if (this.currentWetness[nv]! >= 0.05) {
            const d = Math.abs(this.currentY[nv]! - y);
            if (d > m) m = d;
          }
        }
        if (j + 1 < vh) {
          const nv = vi + vw;
          if (this.currentWetness[nv]! >= 0.05) {
            const d = Math.abs(this.currentY[nv]! - y);
            if (d > m) m = d;
          }
        }
      }
    }
    return m;
  }

  private showFatalBanner(): void {
    const el = (typeof document !== 'undefined' && document.getElementById('err')) as HTMLElement | null;
    if (!el) return;
    el.style.display = 'block';
    el.textContent =
      `WaterSurface: ${this.consecutiveErrors} consecutive GPU readback failures.\n` +
      `Last error: ${this.lastErrorMessage}\n` +
      `(WebGPU device may have been lost; check browser console.)`;
  }
}

type Float32BufferAttribute = THREE.BufferAttribute & { array: Float32Array };
