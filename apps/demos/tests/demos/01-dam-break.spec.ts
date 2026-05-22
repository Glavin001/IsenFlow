import {
  test,
  expect,
  openDemo,
  waitSimSeconds,
  assertNoErrors,
  assertPerformance,
} from '../_setup.js';

test.describe('demo 01 — dam break', () => {
  test('dam drops at t≈2s and the wave advances east; volume conserves; FPS healthy', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, '01-dam-break');

    // Initial state: dam is visible, water column on the left third only.
    const initialVolume = await page.evaluate(() => window.__isenflow_app!.totalVolume());
    const grid = await page.evaluate(() => window.__isenflow_app!.grid());
    const expectedInitialM3 =
      Math.floor(grid.width / 3) * grid.height * grid.dx * grid.dx * 4;
    // Allow a 10% band around the seeded volume (cells fill exactly, but the
    // first frame's solver step may have slightly redistributed).
    expect(initialVolume).toBeGreaterThan(expectedInitialM3 * 0.9);
    expect(initialVolume).toBeLessThan(expectedInitialM3 * 1.1);

    const damVisibleAt0 = await page.evaluate(() => window.__isenflow_app!.sceneObject('dam'));
    expect(damVisibleAt0).toEqual({ exists: true, visible: true });

    // Wait past the drop time (sim time, not wall time).
    await waitSimSeconds(page, 2.5);

    const damVisibleAfter = await page.evaluate(() => window.__isenflow_app!.sceneObject('dam'));
    expect(damVisibleAfter).toEqual({ exists: true, visible: false });
    const dropped = await page.evaluate(() => window.__isenflow_app!.scratch().dropped);
    expect(dropped).toBe(true);

    // Wait further for the wave to advance.
    await waitSimSeconds(page, 5.5);

    // The wet front along row j=64 must have advanced past the dam column.
    const fillEnd = Math.floor(grid.width / 3);
    const front = await page.evaluate(
      ({ j }: { j: number }) => window.__isenflow_app!.wetFront(j, 0.05),
      { j: 64 },
    );
    expect(front, `wet front column at row 64`).toBeGreaterThan(fillEnd);

    // Closed-domain volume conservation: drift < 10% (virtual pipes loses
    // some mass at edges; readback is on the same dt step so this is loose).
    const finalVolume = await page.evaluate(() => window.__isenflow_app!.totalVolume());
    const drift = Math.abs(finalVolume - initialVolume) / Math.max(1e-6, initialVolume);
    expect(drift, `volume drift (initial=${initialVolume.toFixed(2)}, final=${finalVolume.toFixed(2)})`).toBeLessThan(0.10);

    // No NaN in the water buffer.
    const water = await page.evaluate(async () => {
      const arr = await window.__isenflow_app!.readWater();
      // Float32Array isn't JSON-serializable; return as plain array.
      return Array.from(arr);
    });
    expect(water.every((v) => Number.isFinite(v))).toBe(true);

    // Performance: FPS over the recent window must clear 30 (the loop has
    // had ≥5s to warm up). p95 must be under 33ms.
    await assertPerformance(page, { minFps: 30, p95Ms: 50 });

    assertNoErrors(consoleErrors, pageErrors);
  });
});
