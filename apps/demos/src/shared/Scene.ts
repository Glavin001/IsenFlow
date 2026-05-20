/**
 * Shared demo scaffolding: Three.js WebGPURenderer + Rapier world + camera + FPS overlay.
 *
 * Each demo registers a `setup(scene)` and `tick(scene, dt)`; the runner here
 * drives the loop.
 */
import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import RAPIER from '@dimforge/rapier3d-compat';

import {
  acquireGPU,
  SimulationGrid,
  VirtualPipesSolver,
  HeightfieldRasterizer,
  SplashParticleSystem,
  isWebGPUAvailable,
} from 'isenflow';

export interface DemoContext {
  three: typeof THREE;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: WebGPURenderer;
  controls: OrbitControls;
  rapier: typeof RAPIER;
  world: RAPIER.World;
  solver: VirtualPipesSolver;
  rasterizer: HeightfieldRasterizer;
  splashes: SplashParticleSystem;
  bodies: Map<number, RAPIER.RigidBody>;
  scratch: Record<string, unknown>;
  waterMesh?: THREE.Mesh;
  /** CPU mirror of water — refreshed every few frames for forces/splash spawning. */
  waterCpu?: Float32Array;
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

export async function createDemoContext(canvas: HTMLCanvasElement): Promise<DemoContext> {
  if (!isWebGPUAvailable()) {
    throw new Error('WebGPU not available. Try Chrome 121+, Edge, or Safari 26+.');
  }
  const gpu = await acquireGPU();
  const renderer = new WebGPURenderer({ canvas, antialias: true, device: gpu.device } as unknown as ConstructorParameters<typeof WebGPURenderer>[0]);
  await renderer.init();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(new THREE.Color(0x121a2b), 1);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x121a2b, 60, 220);
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(40, 30, 40);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0, 0);
  controls.update();

  await RAPIER.init();
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

  // Default 128×128 grid at 0.5m dx -> 64m × 64m active area.
  const grid = new SimulationGrid({
    width: 128,
    height: 128,
    dx: 0.5,
    origin: [-32, -32],
  });
  const solver = new VirtualPipesSolver(gpu, grid, { dt: 1 / 240, substepsPerFrame: 4 });
  const rasterizer = new HeightfieldRasterizer(solver);
  const splashes = new SplashParticleSystem(4000);

  // Basic lighting.
  const hemi = new THREE.HemisphereLight(0xbfd5ff, 0x223344, 0.7);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(40, 60, 20);
  scene.add(sun);

  // Ground plane visual (water mesh added per demo by sharing CPU water buffer).
  const groundGeom = new THREE.PlaneGeometry(64, 64, 1, 1);
  groundGeom.rotateX(-Math.PI / 2);
  const ground = new THREE.Mesh(groundGeom, new THREE.MeshStandardMaterial({ color: 0x3a3324, roughness: 1 }));
  ground.position.y = -0.01;
  scene.add(ground);

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  return {
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
    bodies: new Map(),
    scratch: {},
  };
}

export async function switchDemo(ctx: DemoContext, demo: Demo): Promise<void> {
  if (activeDemo?.cleanup && activeCtx) activeDemo.cleanup(activeCtx);
  // Hide previous bodies + clear scene of demo-tagged objects
  for (const obj of [...ctx.scene.children]) {
    if ((obj as { userData?: { demoOwned?: boolean } }).userData?.demoOwned) {
      ctx.scene.remove(obj);
    }
  }
  ctx.bodies.clear();
  ctx.scratch = {};
  // Recreate the world (simplest reset)
  ctx.world = new ctx.rapier.World({ x: 0, y: -9.81, z: 0 });
  activeDemo = demo;
  activeCtx = ctx;
  await demo.setup(ctx);
}

export function startLoop(ctx: DemoContext, onFps: (fps: number) => void): void {
  let last = performance.now();
  let frames = 0;
  let fpsT = last;
  const tick = () => {
    requestAnimationFrame(tick);
    const now = performance.now();
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (activeDemo && activeCtx) {
      try {
        ctx.world.step();
        activeDemo.tick(ctx, dt);
        ctx.solver.step();
        ctx.splashes.tick(dt);
        ctx.renderer.render(ctx.scene, ctx.camera);
      } catch (err) {
        showError(err as Error);
      }
    }
    frames++;
    if (now - fpsT > 500) {
      onFps(Math.round((frames * 1000) / (now - fpsT)));
      frames = 0;
      fpsT = now;
    }
  };
  tick();
}

function showError(err: Error): void {
  const el = document.getElementById('err');
  if (!el) return;
  el.style.display = 'block';
  el.textContent = `${err.name}: ${err.message}\n\n${err.stack ?? ''}`;

  console.error(err);
}

/** Convenience: mark an object as demo-owned for clean-up on switch. */
export function ownByDemo<T extends THREE.Object3D>(obj: T): T {
  obj.userData.demoOwned = true;
  return obj;
}
