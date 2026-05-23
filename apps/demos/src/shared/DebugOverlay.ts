/**
 * Debug overlay: color-coded 3D meshes showing the solver's actual
 * bed elevation and water depth. Toggled via button or D key.
 *
 * Wired into the main loop in Scene.ts so every demo gets it for free.
 */
import * as THREE from 'three';
import type { VirtualPipesSolver } from 'isenflow';

const MODE_LABELS = ['OFF', 'BED ELEVATION', 'WATER DEPTH', 'BED + WATER'];

export class DebugOverlay {
  readonly bedMesh: THREE.Mesh;
  readonly waterMesh: THREE.Mesh;
  private bedPositions: THREE.BufferAttribute;
  private bedColors: THREE.BufferAttribute;
  private waterPositions: THREE.BufferAttribute;
  private waterColors: THREE.BufferAttribute;
  private mode = 0;
  private reading = false;
  private framesSinceRead = 0;
  private btn: HTMLButtonElement | null = null;
  private onKeyDown: ((e: KeyboardEvent) => void) | null = null;

  constructor(private readonly solver: VirtualPipesSolver) {
    const g = solver.grid;
    const W = g.width;
    const H = g.height;
    const worldW = W * g.dx;
    const worldH = H * g.dx;
    const centerX = g.origin[0] + worldW / 2;
    const centerZ = g.origin[1] + worldH / 2;

    // Bed elevation mesh (green → yellow → red)
    const bedGeo = new THREE.PlaneGeometry(worldW, worldH, W, H);
    bedGeo.rotateX(-Math.PI / 2);
    this.bedPositions = bedGeo.getAttribute('position') as THREE.BufferAttribute;
    const bedColorArr = new Float32Array((W + 1) * (H + 1) * 3);
    this.bedColors = new THREE.BufferAttribute(bedColorArr, 3);
    bedGeo.setAttribute('color', this.bedColors);
    this.bedMesh = new THREE.Mesh(bedGeo, new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.85,
      side: THREE.DoubleSide, depthWrite: false,
    }));
    this.bedMesh.name = '_debugBed';
    this.bedMesh.position.set(centerX, 0.02, centerZ);
    this.bedMesh.visible = false;
    this.bedMesh.renderOrder = 10;

    // Water depth mesh (grey → cyan → blue)
    const waterGeo = new THREE.PlaneGeometry(worldW, worldH, W, H);
    waterGeo.rotateX(-Math.PI / 2);
    this.waterPositions = waterGeo.getAttribute('position') as THREE.BufferAttribute;
    const waterColorArr = new Float32Array((W + 1) * (H + 1) * 3);
    this.waterColors = new THREE.BufferAttribute(waterColorArr, 3);
    waterGeo.setAttribute('color', this.waterColors);
    this.waterMesh = new THREE.Mesh(waterGeo, new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.75,
      side: THREE.DoubleSide, depthWrite: false,
    }));
    this.waterMesh.name = '_debugWater';
    this.waterMesh.position.set(centerX, 0.04, centerZ);
    this.waterMesh.visible = false;
    this.waterMesh.renderOrder = 11;
  }

  /** Add meshes to scene and install button + keyboard shortcut. */
  install(scene: THREE.Scene): void {
    scene.add(this.bedMesh);
    scene.add(this.waterMesh);

    // Button in HUD
    const hud = document.getElementById('hud');
    if (hud) {
      this.btn = document.createElement('button');
      this.btn.id = 'debug-toggle';
      this.btn.textContent = `Debug: ${MODE_LABELS[0]}`;
      this.btn.style.cssText =
        'width:100%;padding:4px 8px;margin-top:6px;background:#1a2438;color:#0f0;' +
        'border:1px solid #2a3a5a;border-radius:4px;font-family:inherit;font-size:12px;cursor:pointer;';
      hud.appendChild(this.btn);
      this.btn.addEventListener('click', () => this.cycle());
    }

    // D key shortcut
    this.onKeyDown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      if (e.key === 'd' || e.key === 'D') this.cycle();
    };
    window.addEventListener('keydown', this.onKeyDown);
  }

  /** Remove meshes from scene and clean up DOM/listeners. */
  uninstall(scene: THREE.Scene): void {
    scene.remove(this.bedMesh);
    scene.remove(this.waterMesh);
    if (this.btn) { this.btn.remove(); this.btn = null; }
    if (this.onKeyDown) {
      window.removeEventListener('keydown', this.onKeyDown);
      this.onKeyDown = null;
    }
    this.mode = 0;
    this.reading = false;
    this.framesSinceRead = 0;
  }

  /** Cycle through: OFF → BED → WATER → BOTH → OFF. */
  cycle(): void {
    this.mode = (this.mode + 1) % 4;
    this.bedMesh.visible = this.mode === 1 || this.mode === 3;
    this.waterMesh.visible = this.mode === 2 || this.mode === 3;
    this.framesSinceRead = 999; // force immediate readback
    if (this.btn) this.btn.textContent = `Debug: ${MODE_LABELS[this.mode]}`;
  }

  /** Call once per frame from the main loop. Reads GPU data when active. */
  update(): void {
    if (this.mode === 0 || this.reading) return;
    this.framesSinceRead++;
    if (this.framesSinceRead < 5) return;
    this.framesSinceRead = 0;
    this.reading = true;

    const g = this.solver.grid;
    const W = g.width;
    const H = g.height;
    const nx = W + 1;

    Promise.all([this.solver.readBed(), this.solver.readWater()])
      .then(([bedData, waterData]) => {
        if (this.mode === 1 || this.mode === 3) {
          const pos = this.bedPositions;
          const col = this.bedColors;
          for (let j = 0; j <= H; j++) {
            for (let i = 0; i <= W; i++) {
              const ci = Math.min(W - 1, i);
              const cj = Math.min(H - 1, j);
              const idx = (cj * W + ci) * 2;
              const bedTotal = bedData[idx + 1] ?? 0;
              const vi = j * nx + i;
              pos.setY(vi, bedTotal);
              const t = Math.min(bedTotal / 4.0, 1.0);
              if (t < 0.5) {
                const s = t * 2;
                col.setXYZ(vi, s, 0.5 + 0.5 * (1 - s), 0);
              } else {
                const s = (t - 0.5) * 2;
                col.setXYZ(vi, 1, 1 - s, 0);
              }
            }
          }
          pos.needsUpdate = true;
          col.needsUpdate = true;
          this.bedMesh.geometry.computeVertexNormals();
        }

        if (this.mode === 2 || this.mode === 3) {
          const pos = this.waterPositions;
          const col = this.waterColors;
          for (let j = 0; j <= H; j++) {
            for (let i = 0; i <= W; i++) {
              const ci = Math.min(W - 1, i);
              const cj = Math.min(H - 1, j);
              const idx = (cj * W + ci) * 2;
              const h = waterData[idx] ?? 0;
              const bedTotal = bedData[idx + 1] ?? 0;
              const vi = j * nx + i;
              pos.setY(vi, h > 0.001 ? bedTotal + h : bedTotal);
              const depth = Math.min(h / 2.0, 1.0);
              if (h < 0.001) {
                col.setXYZ(vi, 0.3, 0.3, 0.3);
              } else {
                col.setXYZ(vi, 0.1 * (1 - depth), 0.4 + 0.2 * (1 - depth), 0.6 + 0.4 * depth);
              }
            }
          }
          pos.needsUpdate = true;
          col.needsUpdate = true;
          this.waterMesh.geometry.computeVertexNormals();
        }
      })
      .catch(() => { /* ignore readback failures in debug */ })
      .finally(() => { this.reading = false; });
  }
}
