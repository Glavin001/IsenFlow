/**
 * Shared Playwright helpers for demo specs:
 *   - WebGPU adapter probing (fail on `chromium-real-gpu` if software).
 *   - Page navigation that waits for `__isenflow_app.ready === true`.
 *   - Console / pageerror collectors that fail tests on uncaught errors.
 *   - Sim-time and demo-state pollers.
 */
import { expect, test as base, type Page, type TestInfo } from '@playwright/test';

/** Names that indicate a software adapter we don't accept on real-gpu. */
const SOFTWARE_VENDORS = [/swiftshader/i, /lavapipe/i, /microsoft basic/i, /llvmpipe/i];

export interface ProbeResult {
  available: boolean;
  reason?: string;
  adapter?: string;
}

export async function probeAdapter(page: Page): Promise<ProbeResult> {
  return page.evaluate(async (): Promise<ProbeResult> => {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      return { available: false, reason: 'navigator.gpu missing' };
    }
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return { available: false, reason: 'requestAdapter() returned null' };
      const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
      const adapterStr = info
        ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' / ')
        : 'unknown';
      const device = await adapter.requestDevice().catch(() => null);
      if (!device) return { available: false, reason: 'requestDevice returned null', adapter: adapterStr };
      device.destroy?.();
      return { available: true, adapter: adapterStr };
    } catch (err) {
      return { available: false, reason: (err as Error).message };
    }
  });
}

export interface DemoFixture {
  consoleErrors: string[];
  pageErrors: string[];
}

export const test = base.extend<DemoFixture>({
  consoleErrors: async ({ page }, use, testInfo) => {
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    await use(errors);
    if (errors.length) {
      await testInfo.attach('console-errors', {
        body: errors.join('\n'),
        contentType: 'text/plain',
      });
    }
  },
  pageErrors: async ({ page }, use, testInfo) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(`${err.name}: ${err.message}\n${err.stack ?? ''}`));
    await use(errors);
    if (errors.length) {
      await testInfo.attach('page-errors', { body: errors.join('\n\n'), contentType: 'text/plain' });
    }
  },
});

export { expect };

/**
 * Open the demo selector at `#demoId`, wait for the test bridge to be ready,
 * probe the adapter, and (on a real-GPU project) fail the test if the
 * adapter looks like a software fallback.
 */
export async function openDemo(
  page: Page,
  testInfo: TestInfo,
  demoId: string,
  opts: { failOnSoftware?: boolean } = {},
): Promise<{ adapter: string }> {
  await page.goto(`/#${demoId}`);
  await page.waitForFunction(() => window.__isenflow_app?.ready === true, undefined, {
    timeout: 60_000,
  });
  // Wait until the loop has actually run once (tickCount > 0).
  await page.waitForFunction(() => (window.__isenflow_app?.tickCount ?? 0) > 0, undefined, {
    timeout: 60_000,
  });
  const probe = await probeAdapter(page);
  await testInfo.attach('adapter', { body: JSON.stringify(probe, null, 2), contentType: 'application/json' });

  const projectMeta = (testInfo.project.metadata ?? {}) as { gpu?: 'real' | 'software' };
  const failOnSoftware = opts.failOnSoftware ?? projectMeta.gpu === 'real';
  if (failOnSoftware) {
    if (!probe.available) {
      throw new Error(`real-gpu project: WebGPU adapter unavailable (${probe.reason ?? 'unknown'})`);
    }
    const a = (probe.adapter ?? '').toLowerCase();
    for (const sw of SOFTWARE_VENDORS) {
      if (sw.test(a)) {
        throw new Error(`real-gpu project: refusing software adapter "${probe.adapter}"`);
      }
    }
  } else if (!probe.available) {
    test.skip(true, `WebGPU adapter unavailable: ${probe.reason ?? 'unknown'}`);
  }

  // Defensive: make sure the fatal error banner is hidden after init.
  const err = await page.evaluate(() => window.__isenflow_app?.err());
  if (err?.visible) {
    throw new Error(`#err visible at startup: ${err.text}`);
  }
  return { adapter: probe.adapter ?? 'unknown' };
}

/** Wait until the demo's reported sim-time reaches `seconds`. */
export async function waitSimSeconds(page: Page, seconds: number, timeoutMs = 60_000): Promise<void> {
  await page.waitForFunction(
    (s) => (window.__isenflow_app?.simTime ?? 0) >= s,
    seconds,
    { timeout: timeoutMs },
  );
}

/** Throw if `consoleErrors` or `pageErrors` are non-empty. Use at end of tests. */
export function assertNoErrors(consoleErrors: string[], pageErrors: string[]): void {
  // Ignore expected/benign warnings that some adapters / browsers emit.
  const benign = [
    /Three\.WebGPURenderer/i,
    /preferredFormat/i,
    /Failed to load resource.*404/i, // missing favicon
    /favicon/i,
  ];
  const filtered = consoleErrors.filter((e) => !benign.some((b) => b.test(e)));
  if (filtered.length) {
    throw new Error(`Unexpected console errors:\n${filtered.join('\n')}`);
  }
  if (pageErrors.length) {
    throw new Error(`Unexpected page errors:\n${pageErrors.join('\n\n')}`);
  }
}

/** Read the bridge's frameStats and assert FPS / p95 frame time. */
export async function assertPerformance(
  page: Page,
  opts: { minFps?: number; p95Ms?: number; p99Ms?: number; minSamples?: number } = {},
): Promise<{ fps: number; p50: number; p95: number; p99: number; samples: number }> {
  const stats = await page.evaluate(() => {
    const a = window.__isenflow_app;
    if (!a) return null;
    return { fps: a.fps(), ...a.frameStats() };
  });
  if (!stats) throw new Error('test bridge unavailable');
  const minSamples = opts.minSamples ?? 30;
  expect(stats.samples, 'frame samples').toBeGreaterThanOrEqual(minSamples);
  if (opts.minFps !== undefined) {
    expect(stats.fps, `fps >= ${opts.minFps}`).toBeGreaterThanOrEqual(opts.minFps);
  }
  if (opts.p95Ms !== undefined) {
    expect(stats.p95, `p95 frame time <= ${opts.p95Ms}ms`).toBeLessThanOrEqual(opts.p95Ms);
  }
  if (opts.p99Ms !== undefined) {
    expect(stats.p99, `p99 frame time <= ${opts.p99Ms}ms`).toBeLessThanOrEqual(opts.p99Ms);
  }
  return stats;
}

/** Assert that no NaN appears in a Float32Array. */
export function assertNoNaN(arr: Float32Array | number[], label: string): void {
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (typeof v === 'number' && Number.isNaN(v)) {
      throw new Error(`${label}: NaN at index ${i}`);
    }
  }
}
