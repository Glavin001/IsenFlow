import { test, expect } from '@playwright/test';

test.describe('SWE solver — real WebGPU compute', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/test-harness.html');
    await page.waitForFunction(() => window.__isenflow_test?.ready === true, { timeout: 60_000 });
    const hasGPU = await page.evaluate(() => window.__isenflow_test.hasWebGPU());
    test.skip(!hasGPU, 'WebGPU not available in this runner');
  });

  test('lake-at-rest: total volume drifts < 1% over 300 steps', async ({ page }) => {
    const result = await page.evaluate(async () => window.__isenflow_test.runConservation(300));
    expect(result.drift).toBeLessThan(0.01);
  });

  test('dam-break: wave front advances in the right ballpark vs Stoker', async ({ page }) => {
    const result = await page.evaluate(async () => window.__isenflow_test.runDamBreak(400, 2, 256));
    // Virtual pipes is known to underestimate wave-front speed by 30-60 %
    // and to be quite dispersive — the goal here is just to confirm the
    // wave moves toward the dry side at all, not analytical accuracy.
    const ratio = result.frontCellAtEnd / result.expectedCell;
    expect(ratio).toBeGreaterThan(0.2);
    expect(ratio).toBeLessThan(1.8);
  });
});
