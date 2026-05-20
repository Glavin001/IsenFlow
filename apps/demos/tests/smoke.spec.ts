import { test, expect } from '@playwright/test';

const DEMOS = [
  '01-dam-break',
  '02-riverboat',
  '03-sinking-vehicles',
  '04-building-flood',
  '05-tsunami',
  '06-cascading-destruction',
  '07-splash',
];

test('demo selector lists all demos', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#demoSelect option')).toHaveCount(DEMOS.length);
});

for (const id of DEMOS) {
  test(`demo ${id} loads without fatal errors`, async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`/#${id}`);
    // Wait for canvas + scene initialization.
    await expect(page.locator('canvas')).toBeVisible({ timeout: 30_000 });
    // Tolerate WebGPU-unavailable banner (CI may not have a working device).
    const err = page.locator('#err');
    if (await err.isVisible().catch(() => false)) {
      const text = (await err.textContent()) ?? '';
      // Acceptable: WebGPU not available.
      expect(text.toLowerCase()).toContain('webgpu');
      test.skip(true, `WebGPU unavailable in this runner: ${text.slice(0, 200)}`);
    }
    // No uncaught JS errors.
    expect(errors, errors.join('\n')).toHaveLength(0);
  });
}
