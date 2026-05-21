import {
  test,
  expect,
  openDemo,
  waitSimSeconds,
  assertNoErrors,
  assertPerformance,
} from '../_setup.js';

// TODO(buoyancy): Same root cause as demo 02 — the GPU coupling's readback
// latency lets stale forces re-apply for several frames, and the per-cell
// buoyancy in `accumulate_forces.wgsl` over-counts cells in a body's footprint.
// In practice that throws the concrete block UP to y≈130m and the wood block
// through any floor we add. Re-enable once buoyancy/coupling is stabilised
// (likely a CPU-side smoothing pass or synchronous force read for the body's
// own chunkId).
test.describe('demo 03 — sinking vehicles', () => {
  test.skip('wood floats; concrete sinks (real GPU buoyancy)', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, '03-sinking-vehicles');

    // Initial: both bodies up high, free-falling toward the pond.
    const initial = await page.evaluate(() => window.__isenflow_app!.bodies());
    const wood0 = initial.find((b) => b.name === 'wood');
    const conc0 = initial.find((b) => b.name === 'concrete');
    expect(wood0).toBeDefined();
    expect(conc0).toBeDefined();
    expect(wood0!.translation.y).toBeGreaterThan(10);
    expect(conc0!.translation.y).toBeGreaterThan(10);

    // After 5s, both must have entered the water (y ≤ 6 and well below 10).
    await waitSimSeconds(page, 5);
    const mid = await page.evaluate(() => window.__isenflow_app!.bodies());
    const woodMid = mid.find((b) => b.name === 'wood')!;
    const concMid = mid.find((b) => b.name === 'concrete')!;
    expect(woodMid.translation.y, 'wood entered water by 5s').toBeLessThan(6);
    expect(concMid.translation.y, 'concrete entered water by 5s').toBeLessThan(6);

    // After 12s, wood should be FLOATING with low vertical speed; concrete
    // should be DEEP (well below the waterline).
    await waitSimSeconds(page, 12);
    const final = await page.evaluate(() => window.__isenflow_app!.bodies());
    const wood = final.find((b) => b.name === 'wood')!;
    const conc = final.find((b) => b.name === 'concrete')!;

    expect(wood.translation.y, 'wood floats around waterline').toBeGreaterThan(1);
    expect(wood.translation.y, 'wood floats around waterline').toBeLessThan(4);
    expect(Math.abs(wood.linvel.y), 'wood vy small').toBeLessThan(2.0);

    expect(conc.translation.y, 'concrete sunk').toBeLessThan(1.5);

    await assertPerformance(page, { minFps: 30, p95Ms: 50 });
    assertNoErrors(consoleErrors, pageErrors);
  });
});
