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
  test(`demo ${id} loads without fatal errors`, async ({ page }, testInfo) => {
    const errors: string[] = [];
    const consoleLog: string[] = [];
    page.on('pageerror', (e) => errors.push(`${e.name}: ${e.message}\n${e.stack ?? ''}`));
    page.on('console', (msg) => consoleLog.push(`[${msg.type()}] ${msg.text()}`));
    await page.goto(`/#${id}`);
    await expect(page.locator('canvas')).toBeVisible({ timeout: 30_000 });

    // Give the demo a beat to surface any fatal init error.
    await page.waitForTimeout(500);

    // Always attach console + pageerrors for forensics.
    await testInfo.attach('page-console', { body: consoleLog.join('\n'), contentType: 'text/plain' });
    if (errors.length) {
      await testInfo.attach('page-errors', { body: errors.join('\n\n'), contentType: 'text/plain' });
    }

    // Tolerate WebGPU-unavailable banner (CI may not have a working device).
    const err = page.locator('#err');
    if (await err.isVisible().catch(() => false)) {
      const text = (await err.textContent()) ?? '';
      const lower = text.toLowerCase();
      if (lower.includes('webgpu') || lower.includes('adapter') || lower.includes('gpu')) {
        test.skip(true, `WebGPU unavailable: ${text.slice(0, 200)}`);
      }
      throw new Error(`Demo fatal error: ${text.slice(0, 400)}`);
    }
    // No uncaught JS errors.
    expect(errors, errors.join('\n')).toHaveLength(0);
  });
}
