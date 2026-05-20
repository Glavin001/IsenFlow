import { defineConfig, devices } from '@playwright/test';

const isLinux = process.platform === 'linux';
const isMac = process.platform === 'darwin';
const isWin = process.platform === 'win32';

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
// Only the SW project should run in Linux CI (no real GPU available there).
// Real-GPU project runs everywhere by default; users can filter with
// `--project=chromium-real-gpu` or `--project=chromium-software`.
const enableRealGPU = process.env.PW_DISABLE_REAL_GPU !== '1' && !(CI && isLinux);
const enableSoftware = process.env.PW_DISABLE_SOFTWARE === '1' ? false : true;

export default defineConfig({
  testDir: 'tests',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
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
