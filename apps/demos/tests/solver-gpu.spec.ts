import { test, expect } from '@playwright/test';

test.describe('SWE solver — real WebGPU compute', () => {
  test.beforeEach(async ({ page }, testInfo) => {
    page.on('console', (msg) => {
      testInfo
        .attach('console', { body: `[${msg.type()}] ${msg.text()}`, contentType: 'text/plain' })
        .catch(() => {});
    });
    await page.goto('/test-harness.html');
    await page.waitForFunction(() => window.__isenflow_test?.ready === true, { timeout: 60_000 });
    const probe = await page.evaluate(() => window.__isenflow_test!.probeAdapter());
    await testInfo.attach('adapter', { body: JSON.stringify(probe), contentType: 'application/json' });
    test.skip(!probe.available, `WebGPU adapter unavailable: ${probe.reason ?? 'unknown'}`);

    // On real-gpu project, also fail (not skip) if vendor is software.
    const meta = (testInfo.project.metadata ?? {}) as { gpu?: 'real' | 'software' };
    if (meta.gpu === 'real') {
      const a = (probe.adapter ?? '').toLowerCase();
      for (const sw of ['swiftshader', 'lavapipe', 'llvmpipe', 'microsoft basic']) {
        if (a.includes(sw)) {
          throw new Error(`real-gpu project: refusing software adapter "${probe.adapter}"`);
        }
      }
    }
  });

  test('lake-at-rest: total volume drifts < 1% over 300 steps', async ({ page }, testInfo) => {
    const result = await page.evaluate(async () => window.__isenflow_test!.runConservation(300));
    const meta = (testInfo.project.metadata ?? {}) as { gpu?: 'real' | 'software' };
    // Real Apple GPU: tighter tolerance. Software CI: looser.
    const tol = meta.gpu === 'real' ? 0.005 : 0.01;
    expect(result.drift, `volume drift (initial=${result.initial}, final=${result.final})`).toBeLessThan(tol);
  });

  test('dam-break: wave front advances vs Stoker', async ({ page }, testInfo) => {
    const result = await page.evaluate(async () => window.__isenflow_test!.runDamBreak(400, 2, 256));
    const ratio = result.frontCellAtEnd / result.expectedCell;
    const meta = (testInfo.project.metadata ?? {}) as { gpu?: 'real' | 'software' };
    // Virtual pipes is dispersive and underestimates; tighten on real GPU.
    const [lo, hi] = meta.gpu === 'real' ? [0.3, 1.6] : [0.2, 1.8];
    expect(ratio, `front ratio ${result.frontCellAtEnd}/${result.expectedCell.toFixed(2)}`).toBeGreaterThan(lo);
    expect(ratio).toBeLessThan(hi);
  });

  test('force accumulator: a moving chunk yields non-zero force', async ({ page }) => {
    const result = await page.evaluate(async () => window.__isenflow_test!.runForceAccumulator!());
    expect(result.nonZeroChunks, 'at least one chunk has force').toBeGreaterThanOrEqual(1);
    expect(result.peakMagnitude, 'peak force magnitude (N)').toBeGreaterThan(0);
  });

  test('displacement round-trip: bed rise pushes water without losing mass', async ({ page }) => {
    const result = await page.evaluate(async () => window.__isenflow_test!.runDisplacement!());
    expect(result.volumeBefore, 'volume before').toBeGreaterThan(0);
    expect(Math.abs(result.volumeAfter - result.volumeBefore) / result.volumeBefore).toBeLessThan(0.02);
    expect(result.totalHDelta, 'water rose in neighbor cells').toBeGreaterThan(0);
  });
});
