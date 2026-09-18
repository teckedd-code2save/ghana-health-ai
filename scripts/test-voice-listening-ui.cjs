const { chromium, expect } = require('playwright/test');
const assert = require('node:assert/strict');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      await page.goto(process.env.TEST_URL || 'http://127.0.0.1:7864/?__theme=light');
      await page.getByRole('button', { name: /Twi voice comparison/ }).click();
      await expect.poll(() => page.locator('audio').evaluateAll(as => as.some(a => a.duration > 0)), { timeout: 30000 }).toBe(true);
      const first = await page.locator('audio').evaluateAll(as => {
        const a = as.find(a => a.duration > 0);
        return { src: a.currentSrc, duration: a.duration };
      });
      assert.ok(Math.abs(first.duration - 6.32) < 0.02);
      assert.equal(await page.getByRole('listbox', { name: 'Voice', exact: true }).inputValue(), 'CosyVoice A');
      await page.getByRole('listbox', { name: 'Voice', exact: true }).click();
      await page.getByRole('option', { name: 'CosyVoice B', exact: true }).click();
      await expect.poll(() => page.locator('audio').evaluateAll(as => as.some(a => Math.abs(a.duration - 6.68) < 0.02)), { timeout: 30000 }).toBe(true);
      await page.getByRole('radio', { name: 'Correct words', exact: true }).check();
      await page.getByRole('listbox', { name: 'Voice', exact: true }).click();
      await page.getByRole('option', { name: 'Stable Twi 6', exact: true }).click();
      await expect.poll(() => page.locator('audio').evaluateAll((as, previous) => as.some(a => a.duration > 2 && a.duration < 3 && a.currentSrc !== previous), first.src), { timeout: 30000 }).toBe(true);
      assert.equal(await page.getByRole('radio', { name: 'Correct words', exact: true }).isChecked(), false);
      assert.equal(await page.getByRole('textbox', { name: 'Spoken text', exact: true }).inputValue(),
        'Me ho mfa me. Mepɛ sɛ mehu ayaresabea a ɛbɛn me.');
      await page.getByRole('listbox', { name: 'Phrase', exact: true }).click();
      await page.getByRole('option', { name: 'Shopping', exact: true }).click();
      await expect.poll(() => page.locator('audio').evaluateAll(as => as.some(a => a.duration > 3 && a.duration < 4)), { timeout: 30000 }).toBe(true);
      assert.ok((await page.getByRole('textbox', { name: 'Spoken text', exact: true }).inputValue()).includes('Adenta'));
      await page.getByRole('listbox', { name: 'Voice', exact: true }).click();
      assert.equal(await page.getByRole('option', { name: 'CosyVoice A', exact: true }).count(), 0);
      await page.getByRole('option', { name: 'CosyVoice stream', exact: true }).click();
      await expect.poll(() => page.locator('audio').evaluateAll(as => as.some(a => a.duration > 7 && a.duration < 8)), { timeout: 30000 }).toBe(true);
      await page.getByRole('listbox', { name: 'Phrase', exact: true }).click();
      await page.getByRole('option', { name: 'Feeling unwell', exact: true }).click();
      await expect.poll(() => page.getByRole('textbox', { name: 'Spoken text', exact: true }).inputValue()).toContain('ayaresabea');
      await page.getByRole('listbox', { name: 'Voice', exact: true }).click();
      await page.getByRole('option', { name: 'CosyVoice A', exact: true }).click();
      await expect.poll(() => page.locator('audio').evaluateAll(as => as.some(a => Math.abs(a.duration - 6.32) < 0.02)), { timeout: 30000 }).toBe(true);
      await page.getByRole('listbox', { name: 'Phrase', exact: true }).click();
      await page.getByRole('option', { name: 'Shopping', exact: true }).click();
      await expect.poll(() => page.getByRole('listbox', { name: 'Voice', exact: true }).inputValue()).toBe('CosyVoice');
      await expect.poll(() => page.getByRole('textbox', { name: 'Spoken text', exact: true }).inputValue()).toContain('Adenta');
      const dimensions = await page.evaluate(() => ({ content: document.documentElement.scrollWidth, viewport: innerWidth }));
      assert.ok(dimensions.content <= dimensions.viewport, JSON.stringify(dimensions));
      assert.deepEqual(errors, []);
      await page.getByRole('textbox', { name: 'Spoken text', exact: true }).scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(__dirname, '../tmp/twi-voice-comparison', `audition-${viewport.width}.png`) });
      console.log(JSON.stringify({ viewport, audioLoaded: true, ratingsReset: true, overflow: false }));
      await page.close();
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
