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

    // Initial body state: crate spawned 20% in from the west edge of the grid.
    // Compute the expected start from the live grid so the test stays valid
    // across world-size refactors.
    const grid = await page.evaluate(() => window.__isenflow_app!.grid());
    const worldW = grid.width * grid.dx;
    const expectedStartX = grid.origin[0] + worldW * 0.2;
    const initial = await page.evaluate(() => window.__isenflow_app!.bodyByName('crate'));
    expect(initial).not.toBeNull();
    expect(initial!.translation.x, 'crate spawn x ≈ origin + 0.2·worldW').toBeCloseTo(
      expectedStartX,
      1,
    );

    // Sim 25s. The shader-computed drag pushes the crate east.
    test.slow();
    await waitSimSeconds(page, 25);

    const after = await page.evaluate(() => window.__isenflow_app!.bodyByName('crate'));
    expect(after).not.toBeNull();

    // Crate must move east of its start, but stay within the world.
    expect(after!.translation.x, 'crate moved east').toBeGreaterThan(initial!.translation.x + 0.1);
    expect(after!.translation.x, 'crate stays in world').toBeLessThan(grid.origin[0] + worldW);

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
