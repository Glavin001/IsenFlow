import { defineConfig, devices } from '@playwright/test';
import { execSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';

const isLinux = process.platform === 'linux';
const isMac = process.platform === 'darwin';
const isWin = process.platform === 'win32';

/**
 * Detect whether a real (hardware) GPU is available to Chromium on this host.
 *
 * - macOS / Windows: assume yes. Both have a hardware GPU path that Chromium
 *   uses by default (Metal / D3D11). There is no realistic CI scenario where
 *   we'd run on these without a GPU.
 * - Linux: probe DRM render nodes and vulkaninfo. A render node alone is not
 *   enough (CI installs lavapipe which exposes a CPU "GPU"); we also require
 *   that vulkaninfo reports at least one non-llvmpipe / non-swiftshader
 *   device, OR that a real DRI device is exposed via /dev/dri.
 *
 * Manual overrides:
 *   PW_FORCE_GPU=real     → force real-gpu only
 *   PW_FORCE_GPU=software → force software only
 */
function detectRealGPU(): boolean {
  const force = process.env.PW_FORCE_GPU;
  if (force === 'real') return true;
  if (force === 'software') return false;

  if (isMac || isWin) return true;
  if (!isLinux) return true;

  // Linux: inspect /dev/dri. Render nodes only (renderD128+) are present even
  // for software rasterizers like lavapipe, so we additionally check that a
  // *card* node (cardN) exists, which generally indicates a real DRM driver
  // (i915, amdgpu, nvidia, etc.).
  let hasCardNode = false;
  try {
    if (existsSync('/dev/dri')) {
      hasCardNode = readdirSync('/dev/dri').some((n) => /^card\d+$/.test(n));
    }
  } catch {
    /* ignore */
  }

  // Cross-check with vulkaninfo: if the only device is llvmpipe/swiftshader,
  // it's not a real GPU even if /dev/dri/cardN is present.
  let hasNonSoftwareVulkan = false;
  try {
    const out = execSync('vulkaninfo --summary 2>/dev/null', {
      encoding: 'utf8',
      timeout: 5000,
    });
    const deviceLines = out
      .split('\n')
      .filter((l) => /deviceName\s*=/.test(l))
      .map((l) => l.toLowerCase());
    if (deviceLines.length > 0) {
      hasNonSoftwareVulkan = deviceLines.some(
        (l) => !l.includes('llvmpipe') && !l.includes('swiftshader'),
      );
    }
  } catch {
    // vulkaninfo missing → fall back to /dev/dri heuristic alone.
    return hasCardNode;
  }

  return hasCardNode && hasNonSoftwareVulkan;
}

const HAS_REAL_GPU = detectRealGPU();

const COMMON_ARGS = [
  '--enable-unsafe-webgpu',
  '--no-sandbox',
  '--disable-gpu-sandbox',
  // The page reads adapter info; allow it.
  '--enable-features=WebGPUExperimentalFeatures',
];

const REAL_GPU_ARGS = [
  ...COMMON_ARGS,
  ...(isMac
    ? ['--use-angle=metal']
    : isLinux
      ? [
          // Hardware path on Linux when a real GPU is available.
          // Headless Chromium still wants this even with hw GPU.
          '--ignore-gpu-blocklist',
          '--enable-zero-copy',
        ]
      : isWin
        ? ['--use-angle=d3d11']
        : []),
];

const SOFTWARE_ARGS = [
  ...COMMON_ARGS,
  // SwiftShader / Lavapipe path. Linux CI relies on this; on macOS/Windows
  // the project still runs but probably yields the platform's software
  // adapter (Microsoft Basic Render Driver / SwiftShader fallback).
  '--enable-features=Vulkan,UseSkiaRenderer',
  '--use-vulkan=swiftshader',
  '--enable-unsafe-swiftshader',
  '--disable-vulkan-fallback-to-gl-for-testing',
];

const CI = !!process.env.CI;

// Project selection is driven by actual GPU availability, not by platform/CI:
//   - real GPU detected  → run chromium-real-gpu, skip software
//   - no real GPU        → run chromium-software, skip real-gpu
// This means a Linux box with a real GPU still runs the hardware path, and
// macOS/Windows in CI still avoids SwiftShader unless asked.
//
// Manual overrides (in priority order):
//   PW_FORCE_GPU=real|software   → force one project (also affects detection)
//   PW_DISABLE_REAL_GPU=1        → drop the real-gpu project
//   PW_DISABLE_SOFTWARE=1        → drop the software project
//   PW_ENABLE_SOFTWARE=1         → additionally enable software alongside real
const enableRealGPU = HAS_REAL_GPU && process.env.PW_DISABLE_REAL_GPU !== '1';
const enableSoftware =
  process.env.PW_DISABLE_SOFTWARE === '1'
    ? false
    : !HAS_REAL_GPU || process.env.PW_ENABLE_SOFTWARE === '1';

if (!enableRealGPU && !enableSoftware) {
  throw new Error(
    'playwright.config: no projects enabled. Check PW_DISABLE_REAL_GPU / PW_DISABLE_SOFTWARE / PW_FORCE_GPU.',
  );
}

// eslint-disable-next-line no-console
console.log(
  `[playwright.config] platform=${process.platform} hasRealGPU=${HAS_REAL_GPU} ` +
    `projects=${[enableRealGPU && 'chromium-real-gpu', enableSoftware && 'chromium-software']
      .filter(Boolean)
      .join(',')}`,
);

export default defineConfig({
  testDir: 'tests',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  // Demos share a single Vite preview + a WebGPU adapter; parallel workers
  // contend on both, which makes "initial state" assertions racy (page-load
  // is fast but the sim loop runs while other workers are negotiating GPU
  // resources, so by the time we read the bridge several seconds of sim
  // time may have elapsed). Serialize across files for determinism.
  fullyParallel: false,
  workers: 1,
  retries: CI ? 2 : 0,
  reporter: CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: true,
    trace: 'retain-on-failure',
  },
  projects: [
    ...(enableRealGPU
      ? [
          {
            name: 'chromium-real-gpu',
            use: {
              ...devices['Desktop Chrome'],
              channel: 'chromium',
              launchOptions: { args: REAL_GPU_ARGS },
            },
            metadata: { gpu: 'real' as const },
          },
        ]
      : []),
    ...(enableSoftware
      ? [
          {
            name: 'chromium-software',
            use: {
              ...devices['Desktop Chrome'],
              channel: 'chromium',
              launchOptions: { args: SOFTWARE_ARGS },
            },
            metadata: { gpu: 'software' as const },
          },
        ]
      : []),
  ],
  webServer: {
    command: 'pnpm preview',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: true,
    timeout: 60_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
