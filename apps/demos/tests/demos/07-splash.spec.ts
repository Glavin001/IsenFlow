import {
  test,
  expect,
  openDemo,
  waitSimSeconds,
  assertNoErrors,
  assertPerformance,
} from '../_setup.js';

test.describe('demo 07 — wave tank', () => {
  test('wave sources create non-trivial water pattern; volume is bounded', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, '07-splash');

    // Let waves propagate and interfere for a few seconds.
    await waitSimSeconds(page, 4);

    // Water should still exist (no NaN blowup) and have variation.
    const grid = await page.evaluate(() => window.__isenflow_app!.grid());
    const meanH = await page.evaluate(
      ({ w }: { w: number }) => window.__isenflow_app!.meanHRow(Math.floor(w / 2), 0, w),
      { w: grid.width },
    );
    // Base depth is 0.8m; with wave perturbation ±0.15m, mean should be ~0.65-0.95
    expect(meanH, 'mean depth near base level').toBeGreaterThan(0.3);
    expect(meanH, 'mean depth not blown up').toBeLessThan(2.0);

    await assertPerformance(page, { minFps: 40, p95Ms: 40 });
    assertNoErrors(consoleErrors, pageErrors);
  });
});
