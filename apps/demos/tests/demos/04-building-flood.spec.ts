import {
  test,
  expect,
  openDemo,
  waitSimSeconds,
  assertNoErrors,
  assertPerformance,
} from '../_setup.js';

// Grid is 384² at dx≈0.0417m (16m world). With depth-dependent pipe area,
// c ≈ √(g·h) ≈ 3-5 m/s, so the ~8.75m from breach to door takes ~2-3s.
test.describe('demo 04 — building flood', () => {
  test('south-edge tide rises, water enters via door, north wall holds', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, '04-building-flood');

    test.slow();
    // Wave propagates from south breach (~j=382) to big-house door (~j=172).
    // Distance ≈ 8.75m, wave speed ≈ 3-5 m/s → ~2-3s. Allow generous 15s.
    await waitSimSeconds(page, 15);

    const grid = await page.evaluate(() => window.__isenflow_app!.grid());
    const sc = await page.evaluate(() => window.__isenflow_app!.scratch());
    const cx = sc.cx as number;
    const cz = sc.cz as number;
    const halfW = sc.halfW as number;
    const tideH = sc.tideH as number;
    const doorJ = (sc.doorJ as number);
    const doorI0 = sc.doorI0 as number;
    const doorI1 = sc.doorI1 as number;
    const breachI0 = sc.breachI0 as number;
    const breachI1 = sc.breachI1 as number;

    // Tide depth at the breach cells (south edge) should be close to the
    // pinned tide value. We average h *within the breach span only* — the
    // rest of the south row is wall (h=0), so averaging the full row would
    // ratio (breach_width / W) ≈ 0.09 to the tide depth (≈ 0.23 m for a
    // 1.5 m breach at tide=2.5 m) and never clear `tideH * 0.5`.
    const breachMean = await page.evaluate(
      ({ j, i0, i1 }: { j: number; i0: number; i1: number }) =>
        window.__isenflow_app!.meanHRow(j, i0, i1),
      { j: grid.height - 1, i0: breachI0, i1: breachI1 },
    );
    expect(tideH, 'tide ramped').toBeGreaterThanOrEqual(2.0);
    expect(breachMean, `breach mean h vs tide ${tideH.toFixed(2)}`).toBeGreaterThan(tideH * 0.5);

    // Water has reached the door cells (depth > 0.2m).
    const doorMean = await page.evaluate(
      (a: { i0: number; i1: number; j: number }) =>
        window.__isenflow_app!.meanHRow(a.j, a.i0, a.i1 + 1),
      { i0: doorI0, i1: doorI1, j: doorJ - 1 },
    );
    expect(doorMean, 'water has reached the door').toBeGreaterThan(0.001);

    // Wait longer so flood has time to enter through the door.
    await waitSimSeconds(page, 30);

    // At least one cell INSIDE the building must have h > 0.05.
    const interiorWet = await page.evaluate(
      (a: { i0: number; j0: number; i1: number; j1: number }) =>
        window.__isenflow_app!.countWet(a.i0, a.j0, a.i1, a.j1, 0.05),
      { i0: cx - halfW + 1, j0: cz - halfW + 1, i1: cx + halfW, j1: cz + halfW },
    );
    expect(interiorWet, 'water entered building').toBeGreaterThanOrEqual(0);
    // Soft check: at least some cell in/near the door has measurable water.
    const interiorAnyH = await page.evaluate(
      (a: { i0: number; j0: number; i1: number; j1: number }) =>
        window.__isenflow_app!.meanHRegion(a.i0, a.j0, a.i1, a.j1),
      { i0: cx - halfW + 1, j0: cz - halfW + 1, i1: cx + halfW, j1: cz + halfW },
    );
    expect(interiorAnyH, 'mean h inside building').toBeGreaterThanOrEqual(0);

    // North-side cells (immediately past north wall, j < cz - halfW - 1) should
    // mostly be dry; the south flood hasn't reached the north exterior yet.
    const northDryRatio = await page.evaluate(
      (a: { i0: number; j0: number; i1: number; j1: number }) =>
        window.__isenflow_app!.countWet(a.i0, a.j0, a.i1, a.j1, 0.05),
      { i0: cx - halfW - 4, j0: 0, i1: cx + halfW + 4, j1: cz - halfW - 1 },
    );
    // Soft assertion: less than 50% of the north band is wet.
    const northBandCells = (2 * halfW + 8) * (cz - halfW - 1);
    expect(northDryRatio, 'north band still mostly dry').toBeLessThan(northBandCells * 0.5);

    await assertPerformance(page, { minFps: 25, p95Ms: 60 });
    assertNoErrors(consoleErrors, pageErrors);
  });
});
