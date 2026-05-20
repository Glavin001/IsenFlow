import {
  test,
  expect,
  openDemo,
  waitSimSeconds,
  assertNoErrors,
  assertPerformance,
} from '../_setup.js';

test.describe('demo 02 — riverboat', () => {
  test('crate is carried east by water-coupled drag (no manual addForce)', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, '02-riverboat');

    // Initial body state: crate at x≈-20.
    const initial = await page.evaluate(() => window.__isenflow_app!.bodyByName('crate'));
    expect(initial).not.toBeNull();
    expect(initial!.translation.x).toBeGreaterThan(-21);
    expect(initial!.translation.x).toBeLessThan(-19);

    // Sim 25s. The shader-computed drag pushes the crate east. Higher-res
    // grid (256² @ 0.25 m) means the inflow needs more time to develop a
    // coherent eastward flow at the crate's column.
    test.slow();
    await waitSimSeconds(page, 25);

    const after = await page.evaluate(() => window.__isenflow_app!.bodyByName('crate'));
    expect(after).not.toBeNull();

    // Crate must move east of its start, but stay in-bounds (X in [-32, 32]).
    expect(after!.translation.x, 'crate moved east').toBeGreaterThan(initial!.translation.x + 0.1);
    expect(after!.translation.x).toBeLessThan(32);

    // Body remains floating at a sensible Y (above the river floor, below the bank top of 2m).
    expect(after!.translation.y, 'crate y').toBeGreaterThan(-1);
    expect(after!.translation.y).toBeLessThan(3);

    // River must be flowing east in the channel (mid-row, away from banks).
    const meanU = await page.evaluate(async () => {
      const v = await window.__isenflow_app!.readVelocity();
      const g = window.__isenflow_app!.grid();
      let sum = 0;
      let n = 0;
      const j = Math.floor(g.height / 2);
      for (let i = 5; i < g.width - 5; i++) {
        sum += v[(j * g.width + i) * 2] ?? 0;
        n++;
      }
      return n === 0 ? 0 : sum / n;
    });
    expect(meanU, `mean water u velocity along mid-row`).toBeGreaterThan(0.0);

    await assertPerformance(page, { minFps: 30, p95Ms: 50 });
    assertNoErrors(consoleErrors, pageErrors);
  });
});
