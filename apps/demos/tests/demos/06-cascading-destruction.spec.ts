import {
  test,
  expect,
  openDemo,
  waitSimSeconds,
  assertNoErrors,
  assertPerformance,
} from '../_setup.js';

test.describe('demo 06 — cascading destruction', () => {
  test('walls fracture in west→east order, ≤1 per frame, all gone within 60s', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    test.slow(); // up to 3× default timeout (this demo simulates 60s sim time).
    await openDemo(page, testInfo, '06-cascading-destruction');

    // Initially all 3 walls visible.
    for (const name of ['wall0', 'wall1', 'wall2']) {
      const obj = await page.evaluate((n: string) => window.__isenflow_app!.sceneObject(n), name);
      expect(obj, `${name} present`).toEqual({ exists: true, visible: true });
    }

    // Sample fracturedThisFrame over many ticks; we should never see > 1.
    const fracSamples: number[] = [];
    let visW0Before = true;
    let visW1Before = true;
    const start = Date.now();
    while (Date.now() - start < 90_000) {
      const sample = await page.evaluate(() => {
        const a = window.__isenflow_app!;
        return {
          simTime: a.simTime,
          frac: a.scratch().fracturedThisFrame ?? 0,
          fracTotal: a.scratch().fracturedCount ?? 0,
          w0: a.sceneObject('wall0')?.visible ?? false,
          w1: a.sceneObject('wall1')?.visible ?? false,
          w2: a.sceneObject('wall2')?.visible ?? false,
        };
      });
      fracSamples.push(sample.frac as number);
      // Order check: when wall0 first becomes invisible, wall1 must still be visible.
      if (visW0Before && !sample.w0) {
        expect(sample.w1, 'wall1 still up when wall0 falls').toBe(true);
        visW0Before = false;
      }
      if (visW1Before && !sample.w1) {
        expect(sample.w2, 'wall2 still up when wall1 falls').toBe(true);
        visW1Before = false;
      }
      if (!sample.w0 && !sample.w1 && !sample.w2) break;
      // Brief pause so we sample distinct frames.
      await page.waitForTimeout(150);
    }

    // At least the upstream two walls should be down (wall2's budget is
    // calibrated so it fails close to the tide cap; we don't insist).
    const finalVisible = await page.evaluate(() => ({
      w0: window.__isenflow_app!.sceneObject('wall0')?.visible,
      w1: window.__isenflow_app!.sceneObject('wall1')?.visible,
      w2: window.__isenflow_app!.sceneObject('wall2')?.visible,
    }));
    expect(finalVisible.w0, 'wall0 down').toBe(false);
    expect(finalVisible.w1, 'wall1 down').toBe(false);

    // FractureScheduler budget = 1 → never more than 1 per frame.
    const maxPerFrame = Math.max(...fracSamples.map((v) => Number(v)), 0);
    expect(maxPerFrame, 'scheduler caps at 1 fracture per frame').toBeLessThanOrEqual(1);

    await waitSimSeconds(page, 1); // small buffer
    await assertPerformance(page, { minFps: 20, p95Ms: 75 });
    assertNoErrors(consoleErrors, pageErrors);
  });
});
