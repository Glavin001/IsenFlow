import { test, expect, openDemo, assertNoErrors } from './_setup.js';

const DEMOS = [
  '01-dam-break',
  '02-riverboat',
  '03-sinking-vehicles',
  '04-building-flood',
  '05-tsunami',
  '06-cascading-destruction',
  '07-splash',
  '08-impact',
  '09-city-flood',
  '10-mountain-terrain',
];

test('demo selector lists all demos', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#demoSelect option')).toHaveCount(DEMOS.length, { timeout: 60_000 });
});

for (const id of DEMOS) {
  test(`smoke: demo ${id} loads, runs, no fatal errors`, async ({
    page,
    consoleErrors,
    pageErrors,
  }, testInfo) => {
    await openDemo(page, testInfo, id);
    await expect(page.locator('canvas')).toBeVisible({ timeout: 30_000 });
    // Let the loop run for a moment.
    await page.waitForTimeout(800);
    const fps = await page.evaluate(() => window.__isenflow_app!.fps());
    expect(fps, 'fps reported').toBeGreaterThanOrEqual(0);
    const errInfo = await page.evaluate(() => window.__isenflow_app!.err());
    expect(errInfo.visible, `#err banner: ${errInfo.text}`).toBe(false);
    assertNoErrors(consoleErrors, pageErrors);
  });
}
