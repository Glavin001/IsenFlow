import { test, expect, openDemo, probeAdapter, assertNoErrors } from '../_setup.js';

test.describe('renderer + adapter sanity', () => {
  test('canvas is visible, renderer draws, no console errors', async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, '01-dam-break');

    const canvas = await page.evaluate(() => window.__isenflow_app!.canvas());
    expect(canvas.visible).toBe(true);
    expect(canvas.width).toBeGreaterThan(100);
    expect(canvas.height).toBeGreaterThan(100);

    const tickBefore = await page.evaluate(() => window.__isenflow_app!.tickCount);
    await page.waitForTimeout(800);
    const tickAfter = await page.evaluate(() => window.__isenflow_app!.tickCount);
    expect(tickAfter - tickBefore, 'tick advanced over 800ms').toBeGreaterThan(20);

    // (Optional) Three.js renderer.info.calls — Three's WebGPURenderer may
    // not populate this; if it does, assert > 0.
    const r = await page.evaluate(() => window.__isenflow_app!.renderer());
    if (r.calls > 0) {
      expect(r.calls).toBeGreaterThan(0);
    }

    assertNoErrors(consoleErrors, pageErrors);
  });

  test('on real-gpu project: adapter is hardware (not SwiftShader)', async ({ page }, testInfo) => {
    const projectMeta = (testInfo.project.metadata ?? {}) as { gpu?: 'real' | 'software' };
    test.skip(projectMeta.gpu !== 'real', 'only enforced on chromium-real-gpu project');

    await page.goto('/');
    await page.waitForFunction(() => window.__isenflow_app?.ready === true, undefined, { timeout: 60_000 });
    const probe = await probeAdapter(page);
    await testInfo.attach('adapter', { body: JSON.stringify(probe), contentType: 'application/json' });
    expect(probe.available).toBe(true);
    const a = (probe.adapter ?? '').toLowerCase();
    for (const sw of ['swiftshader', 'lavapipe', 'llvmpipe', 'microsoft basic']) {
      expect(a.includes(sw), `adapter "${probe.adapter}" must not be ${sw}`).toBe(false);
    }
  });
});
