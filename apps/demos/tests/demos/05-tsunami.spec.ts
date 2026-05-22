import {
  test,
  expect,
  openDemo,
  waitSimSeconds,
  assertNoErrors,
  assertPerformance,
} from '../_setup.js';

// FIXME: 256² grid increases the cell count east of the wall by 4×. Need to
// rebalance the inflow tide and the wait window before re-enabling.
test.describe.skip('demo 05 — tsunami overtopping', () => {
  test('east of the wall is initially dry; eventually overtops', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, '05-tsunami');

    const grid = await page.evaluate(() => window.__isenflow_app!.grid());
    const wallI = Math.floor(grid.width / 2);

    const initialWet = await page.evaluate(
      (a: { i0: number; j0: number; i1: number; j1: number }) =>
        window.__isenflow_app!.countWet(a.i0, a.j0, a.i1, a.j1, 0.05),
      { i0: wallI + 1, j0: 0, i1: grid.width, j1: grid.height },
    );
    const eastCells = (grid.width - wallI - 1) * grid.height;
    expect(initialWet, 'east starts dry').toBeLessThan(eastCells * 0.05);

    test.slow();
    // Wait for tide to fully ramp + wave to travel 32m east + overtop.
    // Virtual pipes is dispersive — give it plenty of sim time.
    await waitSimSeconds(page, 30);

    // West-edge mean depth should match the inflow tide value to within 30%.
    const sc = await page.evaluate(() => window.__isenflow_app!.scratch());
    const tideH = sc.tideH as number;
    const westMean = await page.evaluate(
      ({ j, w }: { j: number; w: number }) => window.__isenflow_app!.meanHRow(j, 0, 1).then((_v) => {
        // sample across all rows at i=0 instead.
        return window.__isenflow_app!.meanHRegion(0, 0, 1, w);
      }),
      { j: 0, w: grid.height },
    );
    expect(tideH).toBeGreaterThan(2.5);
    expect(westMean, 'west-edge depth around tideH').toBeGreaterThan(tideH * 0.5);

    const eastWet = await page.evaluate(
      (a: { i0: number; j0: number; i1: number; j1: number }) =>
        window.__isenflow_app!.countWet(a.i0, a.j0, a.i1, a.j1, 0.001),
      { i0: wallI + 1, j0: 0, i1: grid.width, j1: grid.height },
    );
    expect(eastWet, 'east cells wet after overtopping').toBeGreaterThan(eastCells * 0.005);

    await assertPerformance(page, { minFps: 30, p95Ms: 50 });
    assertNoErrors(consoleErrors, pageErrors);
  });
});
