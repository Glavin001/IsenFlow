/**
 * Pragmatic water-surface renderer: a plane geometry whose vertices are
 * updated from a CPU-mirror of the water depth texture every few frames.
 *
 * This keeps the demo simple and portable — no TSL graph plumbing required.
 * For production you'd port this to a vertex shader that samples the GPU
 * texture directly (see WaterMaterial in the library).
 */
import * as THREE from 'three';
import type { VirtualPipesSolver } from 'isenflow';

export class WaterSurface {
  readonly mesh: THREE.Mesh;
  readonly geom: THREE.PlaneGeometry;
  private positions: Float32BufferAttribute;

  private framesSinceRead = 0;
  private reading = false;

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
        const g = this.solver.grid;
        const pos = this.positions;
        const nx = g.width + 1;
        for (let j = 0; j <= g.height; j++) {
          for (let i = 0; i <= g.width; i++) {
            const ci = Math.min(g.width - 1, i);
            const cj = Math.min(g.height - 1, j);
            const idx = (cj * g.width + ci) * 2;
            const h = w[idx] ?? 0;
            const bed = b[idx] ?? 0;
            const y = h > 0.005 ? bed + h : -1;
            pos.setY(j * nx + i, y);
          }
        }
        pos.needsUpdate = true;
        this.geom.computeVertexNormals();
      })
      .catch(() => {})
      .finally(() => { this.reading = false; });
  }
}

type Float32BufferAttribute = THREE.BufferAttribute & { array: Float32Array };
