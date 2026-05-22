/**
 * Pragmatic water-surface renderer: a plane geometry whose vertices are
 * updated from a CPU-mirror of the water depth texture every few frames.
 *
 * For production you'd port this to a vertex shader that samples the GPU
 * texture directly (see WaterMaterial in the library).
 */
import * as THREE from 'three';
import type { VirtualPipesSolver } from 'isenflow';

const READBACK_ERROR_THRESHOLD = 5;

export class WaterSurface {
  readonly mesh: THREE.Mesh;
  readonly geom: THREE.PlaneGeometry;
  private positions: Float32BufferAttribute;

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

  constructor(private readonly solver: VirtualPipesSolver) {
    const g = solver.grid;
    this.geom = new THREE.PlaneGeometry(g.width * g.dx, g.height * g.dx, g.width, g.height);
    this.geom.rotateX(-Math.PI / 2);
    this.positions = this.geom.attributes.position as Float32BufferAttribute;
    const mat = new THREE.MeshStandardMaterial({
      color: 0x3870c8,
      transparent: true,
      opacity: 0.85,
      roughness: 0.2,
      metalness: 0.05,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(this.geom, mat);
    this.mesh.name = 'water';
    this.mesh.position.set(
      g.origin[0] + (g.width * g.dx) / 2,
      0,
      g.origin[1] + (g.height * g.dx) / 2,
    );
    this.mesh.frustumCulled = false;
  }

  /** Pull the current water depth from the GPU every ~3 frames. */
  update(): void {
    if (this.reading) return;
    this.framesSinceRead++;
    if (this.framesSinceRead < 3) return;
    this.framesSinceRead = 0;
    this.reading = true;
    Promise.all([this.solver.readWater(), this.solver.readBed()])
      .then(([w, b]) => {
        this.lastWaterData = w;
        this.lastBedData = b;
        const g = this.solver.grid;
        const pos = this.positions;
        const nx = g.width + 1;
        for (let j = 0; j <= g.height; j++) {
          for (let i = 0; i <= g.width; i++) {
            const ci = Math.min(g.width - 1, i);
            const cj = Math.min(g.height - 1, j);
            const idx = (cj * g.width + ci) * 2;
            const h = w[idx] ?? 0;
            // Use total bed (channel .y, idx+1) so dynamic-chunk-elevated
            // cells render correctly. The previous code read the terrain
            // channel which made dynamic obstacles invisible to the surface.
            const bedTotal = b[idx + 1] ?? 0;
            // Dry cells: park the vertex deep below the world so it never
            // shows. Parking at the bed makes the plane climb up walls
            // (their bed is the wall height); parking at y=-1 produced a
            // visible halo at the world edges. y=-1000 is reliably hidden.
            const y = h > 0.005 ? bedTotal + h : -1000;
            pos.setY(j * nx + i, y);
          }
        }
        pos.needsUpdate = true;
        this.geom.computeVertexNormals();
        this.consecutiveErrors = 0;
        this.hasFreshData = true;
      })
      .catch((err) => {
        this.consecutiveErrors++;
        this.lastErrorMessage = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
        // Surface readback failures so they're visible. The previous code
        // silently swallowed them, masking adapter loss / disconnects.
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

  /** Last surfaced error message — useful for tests. */
  get readbackError(): { count: number; message: string } {
    return { count: this.consecutiveErrors, message: this.lastErrorMessage };
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
