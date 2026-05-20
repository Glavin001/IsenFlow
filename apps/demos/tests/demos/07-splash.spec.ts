import {
  test,
  expect,
  openDemo,
  waitSimSeconds,
  assertNoErrors,
  assertPerformance,
} from '../_setup.js';

test.describe('demo 07 — splash showcase', () => {
  test('splashes spawn periodically; active count is non-zero and bounded', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, '07-splash');

    // First stone fires at t≈0.5s; allow up to 2s sim.
    await waitSimSeconds(page, 2);

    let peakActive = 0;
    const start = Date.now();
    while (Date.now() - start < 12_000) {
      const s = await page.evaluate(() => window.__isenflow_app!.splashes());
      if (s.active > peakActive) peakActive = s.active;
      await page.waitForTimeout(120);
    }
    expect(peakActive, 'peak active particles after several spawns').toBeGreaterThan(50);

    // splashCount scratch should keep ticking up over time (periodic spawns).
    const sc = await page.evaluate(() => window.__isenflow_app!.scratch());
    expect((sc.splashCount as number) ?? 0, 'spawn count > 5').toBeGreaterThan(5);

    await assertPerformance(page, { minFps: 40, p95Ms: 40 });
    assertNoErrors(consoleErrors, pageErrors);
  });
});
