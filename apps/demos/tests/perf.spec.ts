import { test, expect, openDemo, waitSimSeconds, assertNoErrors } from './_setup.js';

interface PerfBudget {
  id: string;
  /** Max acceptable p50 frame time (ms). Default 16.7ms (60 FPS). */
  p50?: number;
  /** Max acceptable p95 frame time (ms). Default 33ms (30 FPS). */
  p95?: number;
  /** Max acceptable p99 frame time (ms). Default 50ms. */
  p99?: number;
}

// Budgets are calibrated for an Apple-Metal real GPU. Software adapters are
// expected to fail; that's fine — the perf project should run on real-gpu only.
const PERF_BUDGETS: PerfBudget[] = [
  { id: '01-dam-break',             p50: 20, p95: 40, p99: 60 },
  { id: '02-riverboat',             p50: 20, p95: 40, p99: 60 },
  { id: '03-sinking-vehicles',      p50: 20, p95: 40, p99: 60 },
  { id: '04-building-flood',        p50: 22, p95: 45, p99: 65 },
  { id: '05-tsunami',               p50: 20, p95: 40, p99: 60 },
  { id: '06-cascading-destruction', p50: 25, p95: 50, p99: 80 },
  { id: '07-splash',                p50: 18, p95: 35, p99: 50 },
];

test.describe('performance — frame-time histograms (real-gpu)', () => {
  test.beforeEach(async ({}, testInfo) => {
    const projectMeta = (testInfo.project.metadata ?? {}) as { gpu?: 'real' | 'software' };
    test.skip(projectMeta.gpu === 'software', 'perf budgets only enforced on real-gpu project');
  });

  for (const budget of PERF_BUDGETS) {
    test(`perf ${budget.id} clears budget`, async ({ page, consoleErrors, pageErrors }, testInfo) => {
      await openDemo(page, testInfo, budget.id);
      // Warm up: let the loop run a few seconds before sampling.
      await waitSimSeconds(page, 3);
      // Reset the in-page frame ring by waiting until ~300 fresh samples.
      await page.evaluate(() => {
        // Drain old samples by reading frameStats — there's no public reset,
        // but the ring caps at 600 so 300 fresh frames give a clean window.
      });
      await waitSimSeconds(page, 6);

      const stats = await page.evaluate(() => window.__isenflow_app!.frameStats());
      const adapter = await page.evaluate(() => window.__isenflow_app!.adapter());
      await testInfo.attach('perf', {
        body: JSON.stringify({ id: budget.id, adapter, ...stats }, null, 2),
        contentType: 'application/json',
      });

      expect(stats.samples, 'frame samples').toBeGreaterThanOrEqual(60);
      if (budget.p50 !== undefined) {
        expect(stats.p50, `${budget.id} p50 ≤ ${budget.p50}ms`).toBeLessThanOrEqual(budget.p50);
      }
      if (budget.p95 !== undefined) {
        expect(stats.p95, `${budget.id} p95 ≤ ${budget.p95}ms`).toBeLessThanOrEqual(budget.p95);
      }
      if (budget.p99 !== undefined) {
        expect(stats.p99, `${budget.id} p99 ≤ ${budget.p99}ms`).toBeLessThanOrEqual(budget.p99);
      }
      assertNoErrors(consoleErrors, pageErrors);
    });
  }
});
