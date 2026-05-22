# IsenFlow

**Real-time shallow-water simulation for the web.** WebGPU virtual-pipes SWE solver, two-way Rapier rigid-body coupling, destruction stress accumulator, splash particles. Built on Three.js (WebGPURenderer) and Rapier.js.

This monorepo contains:

| Package | Purpose |
|---|---|
| `packages/isenflow` | The library. Pure-math via `isenflow/math`; full WebGPU surface via `isenflow`. |
| `apps/demos` | Vite single-page app with seven runnable scenarios (dam break, riverboat, sinking vehicles, building flood, tsunami overtopping, cascading destruction, splash showcase). |

## Architecture (in one paragraph)

Heightfield 2.5D Shallow Water Equations solved by virtual pipes (Mei et al. 2007). Terrain, walls, debris and dynamic bodies are all encoded as one unified bed elevation — water flows through doors because that cell's bed is just low. Per-cell forces (buoyancy, hydrostatic, drag, vertical damping) are summed into per-chunk fixed-point atomic accumulators (no WebGPU `atomic<f32>` needed), then read back asynchronously and applied to Rapier bodies with one frame of latency. Body-→-water displacement uses a Kellomäki 2014-style volume-conserving redistribution. Splashes are decorative GPU/CPU particles, not coupled back. The full design rationale lives in the spec the repo was bootstrapped from.

## Quickstart

```bash
pnpm install
pnpm dev            # runs the demo app on http://localhost:5173
```

Open the URL in **Chrome 121+ / Edge / Firefox / Safari 26+** (any browser shipping WebGPU). Use the dropdown in the top-left to switch demos.

The HUD in the top-right reports:

| Field | Meaning |
|---|---|
| **Render** | Render frames per second + median CPU frame cost (ms). |
| **Sim step** | Median CPU encode cost of `solver.step()` (ms) per render frame, and the number of physics substeps run last frame. |
| **Sim Hz** | Effective physics steps per real second (substeps × FPS). |
| **Sim rate** | Sim time advanced per second of wall clock — `1.0×` = real time. |
| **Speed** | Slider that drives the loop's `simSpeed` multiplier (default `2.0×`). |

Default grid is **256 × 256 cells at `dx = 0.25 m`** (a 64 m × 64 m world). Internal solver tick is `1/240 s`; the loop runs as many substeps per render frame as needed to advance roughly `simSpeed × dt_real` of sim time.

### Build & preview

```bash
pnpm build          # builds library + demos
pnpm preview        # serves built demos on :4173
```

### Tests

```bash
pnpm test:unit       # Vitest pure-math (Node, no GPU)
pnpm test:e2e        # Playwright — runs both projects below
pnpm test:e2e:real   # Real-GPU project: Apple Metal on macOS / hardware on Linux
pnpm test:e2e:sw     # Software project: SwiftShader/Lavapipe (Linux CI default)
```

Playwright auto-detects the platform:

| Platform | Real-GPU project flags | Software project flags |
|---|---|---|
| macOS | `--use-angle=metal --enable-unsafe-webgpu` | `--use-vulkan=swiftshader` (yields software fallback) |
| Linux | `--ignore-gpu-blocklist --enable-unsafe-webgpu` | `--use-vulkan=swiftshader --enable-unsafe-swiftshader` |
| Windows | `--use-angle=d3d11 --enable-unsafe-webgpu` | `--use-vulkan=swiftshader` |

The real-GPU project **fails** (does not skip) if the adapter is SwiftShader / Lavapipe / Microsoft Basic Render Driver. In CI:
- Linux runners: software project only (no real GPU).
- macOS runners: real-GPU project (Apple Metal).

The e2e suite asserts:
* every demo loads, runs, has zero console / pageerrors
* per-demo physical invariants (volume conservation, body trajectories, depth-at-row, splash particle counts, FPS)
* `lake-at-rest` volume drift < 0.5 % (real GPU) / 1 % (software) over 300 steps
* dam-break wave-front ratio in [0.3, 1.6] (real) / [0.2, 1.8] (software)
* force-accumulator round-trip — a moving chunk produces non-zero force readback
* displacement round-trip — bed rise pushes water without losing mass
* per-demo p50 / p95 / p99 frame-time budgets (real-GPU project only)

## Using the library

```ts
import {
  acquireGPU, SimulationGrid, VirtualPipesSolver,
  HeightfieldRasterizer, SplashParticleSystem,
} from 'isenflow';

const gpu = await acquireGPU();
const grid = new SimulationGrid({ width: 128, height: 128, dx: 0.5, origin: [-32, -32] });
const solver = new VirtualPipesSolver(gpu, grid, { dt: 1/240, substepsPerFrame: 4 });
const raster = new HeightfieldRasterizer(solver);

raster.bakeBoxObstacle({ aabb: { minX: -2, minZ: -2, maxX: 2, maxZ: 2 }, topY: 3 });

function frame() {
  solver.step();
  requestAnimationFrame(frame);
}
frame();
```

Pure math (no WebGPU, no Three.js, no Rapier dependencies — safe to import in any Node test):

```ts
import { cellBuoyancy, hydrostaticHorizontalForce, damBreakFrontPosition } from 'isenflow/math';
```

## CI & deploys

| Workflow | What it does |
|---|---|
| `.github/workflows/ci.yml` | Vitest, typecheck, build, Playwright (with Mesa Vulkan / SwiftShader for software WebGPU). |
| `vercel.json` | Vercel reads it and deploys `apps/demos/dist` on push. |

Connect the repo on Vercel and pushes will produce preview URLs; the production deploy serves the demos.

## Project structure

```
packages/isenflow/
  src/
    core/           VirtualPipesSolver + WGSL kernels + GPUContext
    coupling/       HeightfieldRasterizer, ForceReadback, RapierBridge, HydrostaticMath
    destruction/    WallChunk, StressAccumulator, FractureScheduler
    particles/      SplashParticleSystem
    render/         WaterMesh, WaterMaterial (TSL)
    boundaries/     BoundaryConditions
    utils/          fixedPoint, gridMath
  tests/unit/       Vitest — pure logic
apps/demos/
  src/
    main.ts         demo selector + loop
    shared/         Scene & WaterSurface
    demos/          7 scenarios
    harness.ts      headless WebGPU entry for Playwright integration tests
  tests/            Playwright specs
```

## Status / scope

What's shipped (spec phases 0-7):
- Virtual-pipes SWE on WebGPU (heightfield 2.5D, unconditionally stable via outflow scaling)
- Unified heightfield bed (terrain + walls + debris in one buffer)
- Pure-math force formulas: buoyancy, hydrostatic, drag, vertical damping
- Per-chunk fixed-point atomic force accumulator + async readback ring
- Rapier bridge (force/torque apply)
- Volume-conserving body→water displacement kernel
- Stress accumulator + rate-limited fracture scheduler
- Splash particle pool (CPU integration, instanced rendering)
- 7 runnable demos
- Vitest unit tests + Playwright/SwiftShader GPU integration tests
- GitHub Actions CI + Vercel deploy config

Deferred (spec phases 8-9):
- Multi-story stacked grids
- Sub-tiling for > 1 km² worlds
- TSL water material (caustics, SSR, advanced foam)
- Adaptive substep count / dirty-tile LOD

## Reference & attribution

WGSL kernels are inspired by `lisyarus/webgpu-shallow-water` (MIT) and the algorithm in
*Mei, Decaudin, Hu (2007) — Fast Hydraulic Erosion Simulation and Visualization on GPU*.
Two-way coupling architecture follows Kellomäki 2014, *Rigid Body Interaction for Large-Scale Real-Time Water Simulation*.

Note: NVIDIA patent **US 8,041,550 B1** covers the displacement-difference redistribution method. The technique is also independently published (Kellomäki). Review patent status before commercial release.

## License

MIT — see `LICENSE`.
