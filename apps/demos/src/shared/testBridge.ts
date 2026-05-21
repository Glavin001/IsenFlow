/**
 * Browser-side bridge that exposes a small introspection API on
 * `window.__isenflow_app` so Playwright tests can assert real physical
 * quantities (volume, body positions, depth-at-row, splash count, fps,
 * adapter info) without poking into the Three.js scene graph.
 */
import type { DemoContext } from './Scene.js';

export interface BodySnapshot {
  chunkId: number;
  name: string | null;
  translation: { x: number; y: number; z: number };
  linvel: { x: number; y: number; z: number };
  enabled: boolean;
}

export interface AppBridge {
  ready: boolean;
  /** Current sim time in seconds (advanced by the loop). */
  simTime: number;
  /** Frame counter (advances every requestAnimationFrame). */
  tickCount: number;
  /** Last computed FPS (rounded). */
  fps(): number;
  /** Quick frame-time stats over the recent ring buffer. */
  frameStats(): { samples: number; meanMs: number; p50: number; p95: number; p99: number; maxMs: number };
  /** Adapter info string (vendor / arch / device). */
  adapter(): string;
  /** Live `<canvas>` info. */
  canvas(): { width: number; height: number; visible: boolean };
  /** Whether the fatal-error banner is visible (and its text). */
  err(): { visible: boolean; text: string };
  /** Solver grid descriptor. */
  grid(): { width: number; height: number; dx: number; origin: [number, number] };
  /** Total water volume (m³). */
  totalVolume(): Promise<number>;
  /** Read full water buffer as Float32Array (length width*height*2 [h, h_prev]). */
  readWater(): Promise<Float32Array>;
  /** Read full bed buffer as Float32Array (length width*height*2 [terrain, total]). */
  readBed(): Promise<Float32Array>;
  /** Read velocity buffer (length width*height*2 [u, v]). */
  readVelocity(): Promise<Float32Array>;
  /** Sample h at a single grid cell. */
  hAt(i: number, j: number): Promise<number>;
  /** Mean h across a row. */
  meanHRow(j: number, iStart?: number, iEnd?: number): Promise<number>;
  /** Mean h across a rectangular region. */
  meanHRegion(i0: number, j0: number, i1: number, j1: number): Promise<number>;
  /** Find rightmost cell along a row whose h > threshold. Returns -1 if none. */
  wetFront(j: number, threshold?: number): Promise<number>;
  /** Count cells in a region with h > threshold. */
  countWet(i0: number, j0: number, i1: number, j1: number, threshold?: number): Promise<number>;
  /** Per-body snapshot. */
  bodies(): BodySnapshot[];
  /** Find a body by `name`. */
  bodyByName(name: string): BodySnapshot | null;
  /** Splash particle stats. */
  splashes(): { active: number; capacity: number };
  /** Renderer stats (drawCalls and triangles). */
  renderer(): { calls: number; triangles: number };
  /** Whether a named scene object exists, with visibility. */
  sceneObject(name: string): { exists: boolean; visible: boolean } | null;
  /** Snapshot of the demo's `scratch` map (only primitive / plain values). */
  scratch(): Record<string, unknown>;
  /**
   * Largest |ΔY| between adjacent wet vertices of the water mesh, in metres.
   * Used to assert the rendered surface has no shoreline cliffs. Returns
   * null if the active demo isn't using a `WaterSurface`.
   */
  waterMaxWetSpike(): number | null;
  /**
   * Largest 1D 2nd-difference along rows/columns of the wet portion of the
   * water mesh — a sensitive checkerboard / "sharp ripple" detector.
   * Returns null if no WaterSurface is active.
   */
  waterMaxRoughness(): number | null;
}

declare global {
  interface Window {
    __isenflow_app?: AppBridge;
  }
}

export function installTestBridge(ctx: DemoContext): void {
  const w = (typeof window !== 'undefined' ? window : undefined);
  if (!w) return;

  function safeScratch(): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(ctx.scratch)) {
      const t = typeof v;
      if (v === null || t === 'string' || t === 'number' || t === 'boolean') {
        out[k] = v;
      } else if (Array.isArray(v) && v.every((x) => ['string', 'number', 'boolean'].includes(typeof x))) {
        out[k] = v.slice();
      }
    }
    return out;
  }

  async function readWater(): Promise<Float32Array> {
    return ctx.solver.readWater();
  }
  async function readBed(): Promise<Float32Array> {
    return ctx.solver.readBed();
  }
  async function readVelocity(): Promise<Float32Array> {
    return ctx.solver.readVelocity();
  }

  function quantile(sorted: number[], q: number): number {
    if (sorted.length === 0) return 0;
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)));
    return sorted[idx]!;
  }

  const bridge: AppBridge = {
    ready: true,
    get simTime() {
      return ctx.simTime;
    },
    get tickCount() {
      return ctx.tickCount;
    },
    fps: () => ctx.lastFps,
    frameStats: () => {
      const samples = ctx.frameTimesMs.slice();
      if (samples.length === 0) {
        return { samples: 0, meanMs: 0, p50: 0, p95: 0, p99: 0, maxMs: 0 };
      }
      const sorted = samples.slice().sort((a, b) => a - b);
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      return {
        samples: samples.length,
        meanMs: mean,
        p50: quantile(sorted, 0.5),
        p95: quantile(sorted, 0.95),
        p99: quantile(sorted, 0.99),
        maxMs: sorted[sorted.length - 1]!,
      };
    },
    adapter: () => ctx.adapterInfo,
    canvas: () => {
      const el = ctx.renderer.domElement as HTMLCanvasElement | null;
      if (!el) return { width: 0, height: 0, visible: false };
      const r = el.getBoundingClientRect();
      const visible = r.width > 0 && r.height > 0 && getComputedStyle(el).display !== 'none';
      return { width: r.width, height: r.height, visible };
    },
    err: () => {
      const el = document.getElementById('err');
      if (!el) return { visible: false, text: '' };
      const visible = getComputedStyle(el).display !== 'none';
      return { visible, text: el.textContent ?? '' };
    },
    grid: () => {
      const g = ctx.solver.grid;
      return { width: g.width, height: g.height, dx: g.dx, origin: [g.origin[0], g.origin[1]] };
    },
    totalVolume: () => ctx.solver.totalVolume(),
    readWater,
    readBed,
    readVelocity,
    hAt: async (i, j) => {
      const water = await readWater();
      const g = ctx.solver.grid;
      if (i < 0 || j < 0 || i >= g.width || j >= g.height) return NaN;
      return water[(j * g.width + i) * 2] ?? 0;
    },
    meanHRow: async (j, iStart = 0, iEnd) => {
      const water = await readWater();
      const g = ctx.solver.grid;
      const i1 = iEnd ?? g.width;
      let sum = 0;
      let n = 0;
      for (let i = iStart; i < i1; i++) {
        sum += Math.max(0, water[(j * g.width + i) * 2] ?? 0);
        n++;
      }
      return n === 0 ? 0 : sum / n;
    },
    meanHRegion: async (i0, j0, i1, j1) => {
      const water = await readWater();
      const g = ctx.solver.grid;
      let sum = 0;
      let n = 0;
      for (let j = j0; j < j1; j++) {
        for (let i = i0; i < i1; i++) {
          sum += Math.max(0, water[(j * g.width + i) * 2] ?? 0);
          n++;
        }
      }
      return n === 0 ? 0 : sum / n;
    },
    wetFront: async (j, threshold = 0.05) => {
      const water = await readWater();
      const g = ctx.solver.grid;
      for (let i = g.width - 1; i >= 0; i--) {
        if ((water[(j * g.width + i) * 2] ?? 0) > threshold) return i;
      }
      return -1;
    },
    countWet: async (i0, j0, i1, j1, threshold = 0.05) => {
      const water = await readWater();
      const g = ctx.solver.grid;
      let n = 0;
      for (let j = j0; j < j1; j++) {
        for (let i = i0; i < i1; i++) {
          if ((water[(j * g.width + i) * 2] ?? 0) > threshold) n++;
        }
      }
      return n;
    },
    bodies: () => {
      const out: BodySnapshot[] = [];
      for (const cb of ctx.coupledBodies.values()) {
        const t = cb.body.translation();
        const v = cb.body.linvel();
        out.push({
          chunkId: cb.chunkId,
          name: cb.name ?? cb.mesh?.name ?? null,
          translation: { x: t.x, y: t.y, z: t.z },
          linvel: { x: v.x, y: v.y, z: v.z },
          enabled: cb.body.isEnabled?.() ?? true,
        });
      }
      return out;
    },
    bodyByName: (name) => {
      for (const cb of ctx.coupledBodies.values()) {
        if (cb.name === name || cb.mesh?.name === name) {
          const t = cb.body.translation();
          const v = cb.body.linvel();
          return {
            chunkId: cb.chunkId,
            name,
            translation: { x: t.x, y: t.y, z: t.z },
            linvel: { x: v.x, y: v.y, z: v.z },
            enabled: cb.body.isEnabled?.() ?? true,
          };
        }
      }
      return null;
    },
    splashes: () => ({ active: ctx.splashes.activeCount(), capacity: ctx.splashes.capacity }),
    renderer: () => {
      const info = (ctx.renderer as unknown as { info?: { render?: { calls?: number; triangles?: number } } }).info;
      return {
        calls: info?.render?.calls ?? 0,
        triangles: info?.render?.triangles ?? 0,
      };
    },
    sceneObject: (name) => {
      const obj = ctx.scene.getObjectByName(name);
      if (!obj) return { exists: false, visible: false };
      return { exists: true, visible: obj.visible };
    },
    scratch: () => safeScratch(),
    waterMaxWetSpike: () => {
      const w = ctx.scratch.water as { maxWetSpike?: number } | undefined;
      return w && typeof w.maxWetSpike === 'number' ? w.maxWetSpike : null;
    },
    waterMaxRoughness: () => {
      const w = ctx.scratch.water as
        | {
            mesh?: { geometry?: { attributes?: Record<string, { array?: Float32Array }> } };
            currentWetness?: Float32Array;
          }
        | undefined;
      // Pull from the mesh's position attribute directly so this works
      // regardless of the WaterSurface internal layout.
      if (!w || !w.mesh?.geometry?.attributes) return null;
      const pos = w.mesh.geometry.attributes.position?.array;
      if (!pos) return null;
      const g = ctx.solver.grid;
      const vw = g.width + 1;
      const vh = g.height + 1;
      let maxAbs2nd = 0;
      // 2nd-difference along rows (skip 1-cell border).
      for (let j = 1; j < vh - 1; j++) {
        for (let i = 1; i < vw - 1; i++) {
          const vi = j * vw + i;
          const yC = pos[vi * 3 + 1]!;
          // Skip the legacy parked-below sentinel range (very large neg Y).
          if (yC < -50) continue;
          const yL = pos[(vi - 1) * 3 + 1]!;
          const yR = pos[(vi + 1) * 3 + 1]!;
          const yD = pos[(vi - vw) * 3 + 1]!;
          const yU = pos[(vi + vw) * 3 + 1]!;
          if (yL < -50 || yR < -50 || yD < -50 || yU < -50) continue;
          const lap = Math.abs(yL + yR + yD + yU - 4 * yC);
          if (lap > maxAbs2nd) maxAbs2nd = lap;
        }
      }
      return maxAbs2nd;
    },
  };

  w.__isenflow_app = bridge;
}

export function updateTestBridge(_ctx: DemoContext): void {
  // The bridge reads ctx live; nothing to do here. Hook is kept in case we
  // later want to publish a per-frame snapshot for cheap polling.
}
