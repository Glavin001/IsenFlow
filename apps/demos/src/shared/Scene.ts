/**
 * Shared demo scaffolding: Three.js WebGPURenderer + Rapier world + camera + FPS overlay.
 *
 * Each demo registers a `setup(scene)` and `tick(scene, dt)`; the runner here
 * drives the loop. The loop wires the full two-way coupling stack
 * (rasterizer → solver.step → accumulateForces → ForceReadback → Rapier).
 */
import * as THREE from 'three';
import { WebGPURenderer, PMREMGenerator } from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SkyMesh } from 'three/examples/jsm/objects/SkyMesh.js';
import RAPIER from '@dimforge/rapier3d-compat';

import {
  acquireGPU,
  SimulationGrid,
  SweSolver,
  HeightfieldRasterizer,
  SplashParticleSystem,
  ForceReadback,
  applyStabilizedForces,
  clampCoupledVelocities,
  createForceSmootherState,
  isWebGPUAvailable,
  type ChunkForce,
  type CoupledBodyInfo,
  type DynamicBodyDescriptor,
  type ForceSmootherState,
} from 'isenflow';

import { installTestBridge, updateTestBridge } from './testBridge.js';

/** A coupled body: a Rapier rigid body + its grid footprint. */
export interface CoupledBody {
  chunkId: number;
  body: RAPIER.RigidBody;
  /** Half-extents along X and Z. (Cuboid colliders only for now.) */
  halfExtentX: number;
  halfExtentZ: number;
  /** Top-Y of the body relative to translation y. (Half-height.) */
  halfExtentY: number;
  /** Reference water level (bed + depth) for CPU buoyancy. Set by demo. */
  waterLevelRef: number;
  /** Reference bed elevation for CPU buoyancy. Set by demo. */
  bedLevelRef: number;
  /** Optional visual mesh, kept in sync with the body in `syncBodyMeshes`. */
  mesh?: THREE.Object3D;
  /** Optional debug name. */
  name?: string;
}

export interface DemoContext {
  three: typeof THREE;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: WebGPURenderer;
  controls: OrbitControls;
  rapier: typeof RAPIER;
  world: RAPIER.World;
  solver: SweSolver;
  rasterizer: HeightfieldRasterizer;
  splashes: SplashParticleSystem;
  forceReadback: ForceReadback;
  /** Map of dynamic-body chunk id → CoupledBody. */
  coupledBodies: Map<number, CoupledBody>;
  /** Legacy alias (kept for any demo that still pokes ctx.bodies). */
  bodies: Map<number, RAPIER.RigidBody>;
  scratch: Record<string, unknown>;
  /** Most recent decoded force vector per chunk, indexed by chunk id. */
  lastForces: ChunkForce[] | null;
  /** Wallclock-second sim time; advanced inside the loop. */
  simTime: number;
  /** Frame counter. */
  tickCount: number;
  /** Last computed FPS. */
  lastFps: number;
  /** Recent per-frame deltas (ms). For perf assertions. */
  frameTimesMs: number[];
  /** Recent per-frame sim-step costs (ms; CPU-encode time for solver.step). */
  simStepMs: number[];
  /** Number of solver substeps performed in the last render frame. */
  lastSubsteps: number;
  /** Sim-time advanced per real second (computed every HUD tick). */
  lastSimRate: number;
  /** Sim-speed multiplier (1.0 = real time). */
  simSpeed: number;
  /** Adapter info string for diagnostics. */
  adapterInfo: string;
  /**
   * Allocate the next chunk id and return a `CoupledBody` registered with
   * the context. Demos call this for every dynamic, water-reactive body.
   */
  spawnCoupledBody(args: {
    name?: string;
    body: RAPIER.RigidBody;
    halfExtents: readonly [number, number, number];
    mesh?: THREE.Object3D;
    waterLevelRef?: number;
    bedLevelRef?: number;
  }): CoupledBody;
}

export interface Demo {
  id: string;
  label: string;
  description: string;
  setup(ctx: DemoContext): void | Promise<void>;
  tick(ctx: DemoContext, dt: number): void;
  /** Optional teardown to release demo-owned resources. */
  cleanup?(ctx: DemoContext): void;
}

let activeDemo: Demo | null = null;
let activeCtx: DemoContext | null = null;
let resetChunkCounter: () => void = () => {};

const MAX_FRAME_SAMPLES = 600;

function buildAdapterInfoString(info: GPUAdapterInfo | null): string {
  if (!info) return 'unknown';
  const parts = [info.vendor, info.architecture, info.device, info.description].filter(
    (p): p is string => !!p,
  );
  return parts.join(' / ') || 'unknown';
}

export async function createDemoContext(canvas: HTMLCanvasElement): Promise<DemoContext> {
  if (!isWebGPUAvailable()) {
    throw new Error('WebGPU not available. Try Chrome 121+, Edge, or Safari 26+.');
  }
  const gpu = await acquireGPU();
  const renderer = new WebGPURenderer({
    canvas,
    antialias: true,
    device: gpu.device,
  } as unknown as ConstructorParameters<typeof WebGPURenderer>[0]);
  await renderer.init();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.35;
  renderer.setClearColor(new THREE.Color(0x86a4c8), 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(14, 10, 14);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0, 0);
  controls.update();

  await RAPIER.init();
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

  // World is 16 m × 16 m (smaller, more dramatic scale: a 4 m building now
  // dominates the frame). 384² cells → dx ≈ 4.17 cm — fine enough that
  // wall/building boundaries are crisp.
  const WORLD = 16;
  const grid = new SimulationGrid({
    width: 384,
    height: 384,
    dx: WORLD / 384,
    origin: [-WORLD / 2, -WORLD / 2],
  });
  // Internal physics tick is 1/960 s with 4 substeps per render frame
  // (240 Hz effective sim rate, same as before).  With dx ≈ 4 cm and h
  // up to ~4 m typical (~8 m transient), max wave speed c = √(g·h) ≈
  // 8.9 m/s → CFL = c·dt/dx ≈ 0.23 — well under the KP central-upwind
  // stability limit of 0.5 (per substep, even before RK2's larger margin).
  // The original dt=1/240 ran at CFL ≈ 0.65 which is marginal for KP and
  // diverged on any transient that briefly pushed h above ~10 m.
  const solver = new SweSolver(gpu, grid, {
    dt: 1 / 960,
    substepsPerFrame: 4,
    manningN: 0.03,
    desingEpsilon: 1e-3,
  });
  const rasterizer = new HeightfieldRasterizer(solver);
  const splashes = new SplashParticleSystem(4000);
  const forceReadback = new ForceReadback(gpu.device, solver.maxChunks);

  const hemi = new THREE.HemisphereLight(0xbfd5ff, 0x223344, 0.7);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);

  // Procedural sky + PMREM environment for water reflections.
  const sky = new SkyMesh();
  sky.scale.setScalar(10000);
  sky.turbidity.value = 10;
  sky.rayleigh.value = 2;
  sky.mieCoefficient.value = 0.005;
  sky.mieDirectionalG.value = 0.8;
  scene.add(sky);

  const sunDirection = new THREE.Vector3();
  const sunPosition = new THREE.Vector3();
  const phi = THREE.MathUtils.degToRad(90 - 8); // elevation 8°
  const theta = THREE.MathUtils.degToRad(135); // azimuth 135°
  sunPosition.setFromSphericalCoords(1, phi, theta);
  sky.sunPosition.value.copy(sunPosition);
  sunDirection.copy(sunPosition).normalize();

  // Align the existing directional light with the sky's sun so shadows match.
  sun.position.copy(sunDirection).multiplyScalar(80);
  scene.add(sun);

  // Expose sun for the water material (picked up via scene.userData).
  scene.userData.sun = sunDirection;

  // PMREM-convolve the sky into an env map. We do this once at startup
  // using a dedicated scene containing only the sky, so geometry added
  // later doesn't bleed into the environment.
  const pmrem = new PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  const skyForEnv = new SkyMesh();
  skyForEnv.scale.setScalar(10000);
  skyForEnv.turbidity.value = sky.turbidity.value;
  skyForEnv.rayleigh.value = sky.rayleigh.value;
  skyForEnv.mieCoefficient.value = sky.mieCoefficient.value;
  skyForEnv.mieDirectionalG.value = sky.mieDirectionalG.value;
  skyForEnv.sunPosition.value.copy(sunPosition);
  envScene.add(skyForEnv);
  // fromScene is async on WebGPU until the backend is ready; the helper
  // returns a render target whose .texture we plug into scene.environment.
  pmrem.fromSceneAsync(envScene).then((rt) => {
    scene.environment = rt.texture;
    scene.background = rt.texture;
    // Sky is baked — remove the live mesh so it doesn't run atmospheric
    // scattering per pixel every frame.
    scene.remove(sky);
  });

  const groundGeom = new THREE.PlaneGeometry(WORLD, WORLD, 1, 1);
  groundGeom.rotateX(-Math.PI / 2);
  const ground = new THREE.Mesh(
    groundGeom,
    new THREE.MeshStandardMaterial({ color: 0x3a3324, roughness: 1 }),
  );
  ground.position.y = -0.01;
  ground.name = 'ground';
  scene.add(ground);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const coupledBodies = new Map<number, CoupledBody>();
  let nextChunkId = 1;
  resetChunkCounter = () => { nextChunkId = 1; };

  const ctx: DemoContext = {
    three: THREE,
    scene,
    camera,
    renderer,
    controls,
    rapier: RAPIER,
    world,
    solver,
    rasterizer,
    splashes,
    forceReadback,
    coupledBodies,
    bodies: new Map(),
    scratch: {},
    lastForces: null,
    simTime: 0,
    tickCount: 0,
    lastFps: 0,
    frameTimesMs: [],
    simStepMs: [],
    lastSubsteps: 0,
    lastSimRate: 0,
    simSpeed: 1,
    adapterInfo: buildAdapterInfoString(gpu.adapterInfo),
    spawnCoupledBody({ name, body, halfExtents, mesh, waterLevelRef, bedLevelRef }) {
      if (nextChunkId >= solver.maxChunks) {
        throw new Error(`spawnCoupledBody: chunk id pool exhausted (>= ${solver.maxChunks})`);
      }
      const id = nextChunkId++;
      const coupled: CoupledBody = {
        chunkId: id,
        body,
        halfExtentX: halfExtents[0],
        halfExtentY: halfExtents[1],
        halfExtentZ: halfExtents[2],
        waterLevelRef: waterLevelRef ?? 1.0,
        bedLevelRef: bedLevelRef ?? 0,
        ...(mesh !== undefined ? { mesh } : {}),
        ...(name !== undefined ? { name } : {}),
      };
      coupledBodies.set(id, coupled);
      ctx.bodies.set(id, body);
      if (mesh && name && !mesh.name) mesh.name = name;
      // Enable CPU buoyancy mode when bodies are present
      solver.setCpuBuoyancyMode(true);
      return coupled;
    },
  };

  installTestBridge(ctx);
  return ctx;
}

export async function switchDemo(ctx: DemoContext, demo: Demo): Promise<void> {
  if (activeDemo?.cleanup && activeCtx) activeDemo.cleanup(activeCtx);

  // Remove all demo-owned scene objects.
  for (const obj of [...ctx.scene.children]) {
    if ((obj as { userData?: { demoOwned?: boolean } }).userData?.demoOwned) {
      ctx.scene.remove(obj);
    }
  }

  // Reset coupling state.
  ctx.bodies.clear();
  ctx.coupledBodies.clear();
  resetChunkCounter();

  // Reset bookkeeping.
  ctx.scratch = {};
  ctx.lastForces = null;
  ctx.simTime = 0;
  ctx.tickCount = 0;
  ctx.frameTimesMs.length = 0;
  ctx.simStepMs.length = 0;

  // Reset solver buffers: bed, water, flux, velocity, chunks, boundary.
  ctx.solver.resetSimState();
  ctx.rasterizer.resetDynamicBodies();
  const cells = ctx.solver.grid.cells;
  ctx.solver.writeBedFull(new Float32Array(ctx.solver.grid.bedSeed));
  const init = ctx.solver.grid.initialDepth;
  const waterReset = new Float32Array(cells);
  if (init > 0) waterReset.fill(init);
  ctx.solver.writeWaterFull(waterReset);
  ctx.solver.writeBoundaryRegionTarget(
    { x: 0, y: 0, w: ctx.solver.grid.width, h: ctx.solver.grid.height },
    0,
    0,
  );

  // Reset particles.
  ctx.splashes.reset();

  // Fresh physics world.
  ctx.world = new ctx.rapier.World({ x: 0, y: -9.81, z: 0 });

  // Stash the ctx so cleanup() can find it if we need to abort, but DON'T
  // expose the new demo to the animation loop until setup completes — the
  // loop would otherwise call demo.tick() against an empty scratch and
  // crash with "Cannot read properties of undefined (reading 'update')".
  activeDemo = null;
  activeCtx = ctx;
  await demo.setup(ctx);
  activeDemo = demo;
}

/** Build a per-frame snapshot of all coupled bodies for the rasterizer. */
function collectBodyDescriptors(coupled: Map<number, CoupledBody>): DynamicBodyDescriptor[] {
  const out: DynamicBodyDescriptor[] = [];
  for (const cb of coupled.values()) {
    if (!cb.body.isEnabled || !cb.body.isEnabled()) continue;
    const t = cb.body.translation();
    const v = cb.body.linvel();
    out.push({
      chunkId: cb.chunkId,
      aabb: {
        minX: t.x - cb.halfExtentX,
        minZ: t.z - cb.halfExtentZ,
        maxX: t.x + cb.halfExtentX,
        maxZ: t.z + cb.halfExtentZ,
      },
      topY: t.y + cb.halfExtentY,
      halfY: cb.halfExtentY,
      linvel: [v.x, v.y, v.z],
      com: [t.x, t.y, t.z],
    });
  }
  return out;
}

function syncMeshes(ctx: DemoContext): void {
  for (const cb of ctx.coupledBodies.values()) {
    if (!cb.mesh) continue;
    const t = cb.body.translation();
    const q = cb.body.rotation();
    cb.mesh.position.set(t.x, t.y, t.z);
    if ('quaternion' in cb.mesh) {
      (cb.mesh as THREE.Object3D & { quaternion: THREE.Quaternion }).quaternion.set(q.x, q.y, q.z, q.w);
    }
  }
}

export interface LoopStats {
  /** Render frames per second (sampled every ~500 ms). */
  fps: number;
  /** Median CPU cost of a render frame in ms (last ~1 s). */
  frameMsP50: number;
  /** Median CPU cost of solver-step encode in ms (last ~1 s). */
  simStepMsP50: number;
  /** Substeps run last frame. */
  substeps: number;
  /** Effective sim-Hz (substeps × sim_dt_per_substep × fps in steps/s). */
  simHz: number;
  /** Sim time advanced per second of wall clock. */
  simRate: number;
  /** Sim-speed multiplier in effect. */
  simSpeed: number;
}

const MAX_SUBSTEPS = 32;

export function startLoop(ctx: DemoContext, onStats: (stats: LoopStats) => void): void {
  let last = performance.now();
  let frames = 0;
  let fpsT = last;
  let simAdvancedThisWindow = 0;
  let substepsThisWindow = 0;
  let firstError: Error | null = null;
  const forceSmoother = createForceSmootherState();

  const dtSub = ctx.solver.opts.dt;

  const loop = () => {
    requestAnimationFrame(loop);
    const now = performance.now();
    const rawDt = (now - last) / 1000;
    const dt = Math.min(0.05, rawDt);
    last = now;
    if (firstError) return; // bail out — fatal banner shown
    if (!activeDemo || !activeCtx) return;

    try {
      ctx.tickCount += 1;

      // Adaptive substepping: advance roughly `simSpeed × dt` of sim time
      // per render frame, never more than MAX_SUBSTEPS sub-ticks.
      const target = Math.max(0, dt) * Math.max(0, ctx.simSpeed);
      const subs = ctx.simSpeed <= 0 ? 0 : Math.min(MAX_SUBSTEPS, Math.max(1, Math.floor(target / dtSub) || 1));
      const simAdvance = subs * dtSub;
      ctx.simTime += simAdvance;
      ctx.lastSubsteps = subs;
      simAdvancedThisWindow += simAdvance;
      substepsThisWindow += subs;

      // 1) Rasterize current frame's dynamic bodies into solver buffers.
      ctx.rasterizer.bakeDynamicBodies(collectBodyDescriptors(ctx.coupledBodies));

      // 2) SWE step (one or more substeps; solver.opts.substepsPerFrame is 1).
      const tSim0 = performance.now();
      for (let s = 0; s < subs; s++) ctx.solver.step();
      const simMs = performance.now() - tSim0;
      ctx.simStepMs.push(simMs);
      if (ctx.simStepMs.length > MAX_FRAME_SAMPLES) ctx.simStepMs.shift();

      // DEBUG: Diagnostic logging — every 10 frames for the first 100
      // (~1.5 s, captures spike onset), then every 60 frames after.
      // Reports max(h), max(|u|), max(|v|), and NaN detection.
      // Silence: set `window.__isenflow_debug = false` in the console.
      const debugEnabled = (window as { __isenflow_debug?: boolean }).__isenflow_debug !== false;
      const shouldDiagnose = debugEnabled && (
        (ctx.tickCount <= 100 && ctx.tickCount % 10 === 0) ||
        (ctx.tickCount > 100 && ctx.tickCount % 60 === 0)
      );
      if (shouldDiagnose) {
        void diagnose(ctx);
      }

      // 3) Body→water displacement (uses prevBed → bed delta).
      ctx.solver.applyDisplacement();

      // 4) Water→body forces (zero, accumulate, copy into readback ring).
      ctx.solver.zeroForces();
      const ringSlot = ctx.forceReadback.acquireWriteSlot();
      ctx.solver.accumulateForces(ringSlot ? { copyTo: ringSlot } : undefined);
      const forces = ctx.forceReadback.poll();
      if (forces) ctx.lastForces = forces;
      let bodyInfos: Map<number, CoupledBodyInfo> | null = null;
      if (ctx.coupledBodies.size > 0) {
        bodyInfos = new Map<number, CoupledBodyInfo>();
        for (const cb of ctx.coupledBodies.values()) {
          bodyInfos.set(cb.chunkId, {
            chunkId: cb.chunkId,
            body: cb.body,
            halfExtents: [cb.halfExtentX, cb.halfExtentY, cb.halfExtentZ],
            waterLevelRef: cb.waterLevelRef,
            bedLevelRef: cb.bedLevelRef,
          });
        }
        applyStabilizedForces(bodyInfos, ctx.lastForces, forceSmoother, ctx.solver.grid.dx);
      }

      // 4b) Advance water shader animation clock by sim time.
      const water = ctx.scratch.water as { advanceTime?: (dt: number) => void } | undefined;
      if (water?.advanceTime) water.advanceTime(simAdvance);

      // 5) Demo-specific update.
      activeDemo.tick(ctx, simAdvance);

      // 6) Step Rapier AFTER all forces have been applied this frame.
      ctx.world.step();

      // 6b) Clamp coupled body velocities AFTER Rapier step (safety net).
      if (bodyInfos) {
        clampCoupledVelocities(bodyInfos);
      }

      // 7) Sync visual meshes to physics state.
      syncMeshes(ctx);

      // 8) Particles + render.
      ctx.splashes.tick(dt);
      ctx.renderer.render(ctx.scene, ctx.camera);

      // 9) Bookkeeping.
      const frameMs = (performance.now() - now) || rawDt * 1000;
      ctx.frameTimesMs.push(frameMs);
      if (ctx.frameTimesMs.length > MAX_FRAME_SAMPLES) ctx.frameTimesMs.shift();

      updateTestBridge(ctx);
    } catch (err) {
      firstError = err as Error;
      showError(err as Error);
    }

    frames++;
    if (now - fpsT > 500) {
      const elapsed = (now - fpsT) / 1000;
      const fps = Math.round(frames / elapsed);
      ctx.lastFps = fps;
      const simRate = simAdvancedThisWindow / elapsed;
      ctx.lastSimRate = simRate;
      const simHz = substepsThisWindow / elapsed;
      onStats({
        fps,
        frameMsP50: percentile(ctx.frameTimesMs, 0.5),
        simStepMsP50: percentile(ctx.simStepMs, 0.5),
        substeps: ctx.lastSubsteps,
        simHz,
        simRate,
        simSpeed: ctx.simSpeed,
      });
      frames = 0;
      fpsT = now;
      simAdvancedThisWindow = 0;
      substepsThisWindow = 0;
    }
  };
  loop();
}

function percentile(arr: readonly number[], p: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * sorted.length)));
  return sorted[idx]!;
}

/**
 * Diagnostic readback — reads the conservative state (h, hu, hv) once and
 * prints max-magnitude metrics + NaN/Inf detection to the console.  Async
 * (uses GPU readback) and fire-and-forget — should not block the main loop.
 *
 * Enable/disable via `window.__isenflow_debug` (default ON).
 *
 * Output format:
 *   [isenflow t=1.23 tick=72] maxH=1.32 maxU=0.85 maxV=0.41 vol=4.32  ← healthy
 *   [isenflow t=1.23 tick=72] !! NaN at idx=12345 (h=NaN, hu=...)    ← BROKEN
 *   [isenflow t=1.23 tick=72] !! SPIKE maxH=12.45 (> 5×expected)      ← spike
 */
let diagnoseInFlight = false;
async function diagnose(ctx: DemoContext): Promise<void> {
  if (diagnoseInFlight) return;
  diagnoseInFlight = true;
  try {
    const solver = ctx.solver as { readState?: () => Promise<Float32Array>; readVelocity: () => Promise<Float32Array>; readWater: () => Promise<Float32Array> };
    // Prefer the KP-native readState() (h, hu, hv per cell) when available.
    let h = 0, hu = 0, hv = 0, maxH = 0, maxAbsHu = 0, maxAbsHv = 0;
    let nanIdx = -1;
    let totalH = 0;
    let n = 0;
    if (solver.readState) {
      const s = await solver.readState();
      n = s.length / 4;
      for (let i = 0; i < n; i++) {
        h = s[i * 4 + 0]!;
        hu = s[i * 4 + 1]!;
        hv = s[i * 4 + 2]!;
        if (!Number.isFinite(h) || !Number.isFinite(hu) || !Number.isFinite(hv)) {
          if (nanIdx < 0) nanIdx = i;
          continue;
        }
        if (h > maxH) maxH = h;
        if (Math.abs(hu) > maxAbsHu) maxAbsHu = Math.abs(hu);
        if (Math.abs(hv) > maxAbsHv) maxAbsHv = Math.abs(hv);
        totalH += h;
      }
    } else {
      const w = await solver.readWater();
      n = w.length / 2;
      for (let i = 0; i < n; i++) {
        h = w[i * 2]!;
        if (!Number.isFinite(h)) { if (nanIdx < 0) nanIdx = i; continue; }
        if (h > maxH) maxH = h;
        totalH += h;
      }
    }
    const avgH = totalH / Math.max(1, n);
    const tag = `[isenflow t=${ctx.simTime.toFixed(2)} tick=${ctx.tickCount}]`;
    if (nanIdx >= 0) {
      // eslint-disable-next-line no-console
      console.error(`${tag} !! NaN at idx=${nanIdx} (h=${h}, hu=${hu}, hv=${hv})`);
      return;
    }
    if (maxH > 10) {
      // eslint-disable-next-line no-console
      console.warn(`${tag} !! SPIKE maxH=${maxH.toFixed(2)} maxHu=${maxAbsHu.toFixed(2)} maxHv=${maxAbsHv.toFixed(2)} avgH=${avgH.toFixed(2)}`);
      return;
    }
    // eslint-disable-next-line no-console
    console.log(`${tag} maxH=${maxH.toFixed(3)} maxHu=${maxAbsHu.toFixed(3)} maxHv=${maxAbsHv.toFixed(3)} avgH=${avgH.toFixed(3)}`);
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[isenflow] diagnose readback failed:', e);
  } finally {
    diagnoseInFlight = false;
  }
}

function showError(err: Error): void {
  const el = document.getElementById('err');
  if (!el) return;
  el.style.display = 'block';
  el.textContent = `${err.name}: ${err.message}\n\n${err.stack ?? ''}`;
  // eslint-disable-next-line no-console
  console.error(err);
}

/** Convenience: mark an object as demo-owned for clean-up on switch. */
export function ownByDemo<T extends THREE.Object3D>(obj: T): T {
  obj.userData.demoOwned = true;
  return obj;
}
