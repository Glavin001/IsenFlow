import { defineConfig, devices } from '@playwright/test';

const WEBGPU_FLAGS = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan,UseSkiaRenderer',
  '--use-vulkan=swiftshader',
  '--enable-unsafe-swiftshader',
  '--disable-gpu-sandbox',
  '--no-sandbox',
];

export default defineConfig({
  testDir: 'tests',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: true,
    launchOptions: { args: WEBGPU_FLAGS },
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium-webgpu',
      use: { ...devices['Desktop Chrome'], channel: 'chromium' },
    },
  ],
  webServer: {
    command: 'pnpm preview',
    port: 4173,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
