/* eslint-disable @typescript-eslint/no-require-imports -- Node CommonJS operational harness. */
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const { readFileSync, mkdirSync } = require('node:fs');
const { chromium } = require('playwright');

// Read only the dedicated review credential; never print or persist it locally.
const ssh = command => execFileSync('ssh', ['-i', '/Users/welcome/.ssh/optimi_github_actions', '-o', 'BatchMode=yes',
  'root@128.140.12.62', command], { encoding: 'utf8' }).trim();
const password = ssh(`python3 -c "from pathlib import Path; print(next(line.partition('=')[2] for line in Path('/opt/ghana-corpus-review/.env').read_text().splitlines() if line.startswith('REVIEW_PASSWORD=')))"`);

async function main() {
  const base = 'https://ghanahealth.serendepify.com';
  const authorization = 'Basic ' + Buffer.from('reviewer:' + password).toString('base64');
  for (const suffix of ['/', '/app.js', '/api', '/progress', '/progress.css']) assert.equal((await fetch(base + '/research/corpus' + suffix)).status, 401);
  for (const collection of ['meaning', 'language', 'conversation', 'health']) {
    const response = await fetch(base + '/research/corpus/api?release=twi-stage-v1-20260911&offset=0&collection=' + collection, { headers: { authorization } });
    const data = await response.json();
    assert.equal(response.status, 200); assert.equal(data.ready, true); assert.ok(data.rows[0].source_hash);
  }
  if (process.env.CORPUS_REFERENCE_REVIEW) {
    const expected = JSON.parse(readFileSync(process.env.CORPUS_REFERENCE_REVIEW, 'utf8'));
    for (let offset = 0; offset < expected.length; offset++) {
      const response = await fetch(base + '/research/corpus/api?release=twi-stage-v1-20260911&collection=teacher&offset=' + offset, { headers: { authorization } });
      const data = await response.json();
      assert.equal(data.total, expected.length);
      const row = data.rows[0];
      assert.equal(row.id, expected[offset].id);
      assert.equal(row.source_hash, expected[offset].source_hash);
      assert.equal(row.reference_analysis.id, expected[offset].reference_analysis.id);
      assert.equal(row.training_eligible, false);
      assert.equal(row.split, 'validation');
    }
  }
  if (process.env.CORPUS_RESPONSE_REVIEW) {
    const expected = JSON.parse(readFileSync(process.env.CORPUS_RESPONSE_REVIEW, 'utf8'));
    // The remote queue checksum covers every byte; exercise representative API
    // rows as well, including the newest targets and pending boundary.
    const proposals = expected.flatMap((row, offset) => row.response.proposed ? [offset] : []);
    const offsets = [0, Math.floor(expected.length / 2), ...proposals.slice(-10),
      expected.findIndex(row => !row.response.proposed), expected.length - 1];
    const remoteHash = ssh('sha256sum /opt/ghana-corpus-review/current/tmp/corpus-response-review/twi-stage-v1-20260911/queue.json').split(/\s+/)[0];
    assert.equal(remoteHash, createHash('sha256').update(readFileSync(process.env.CORPUS_RESPONSE_REVIEW)).digest('hex'));
    for (const offset of new Set(offsets.filter(offset => offset >= 0))) {
      const response = await fetch(base + '/research/corpus/api?release=twi-stage-v1-20260911&collection=responses&offset=' + offset, { headers: { authorization } });
      const data = await response.json();
      assert.equal(response.status, 200);
      assert.equal(data.total, expected.length);
      const row = { ...data.rows[0] }; delete row.review;
      assert.deepEqual(row, expected[offset]);
    }
  }
  const html = await (await fetch(base + '/research/corpus/', { headers: { authorization } })).text();
  const names = [...html.matchAll(/app-[a-f0-9]{12}\.(?:js|css)/g)].map(match => match[0]);
  assert.equal(names.length, 2);
  for (const name of names) {
    const response = await fetch(base + '/research/corpus/' + name, { headers: { authorization } });
    const digest = value => createHash('sha256').update(value).digest('hex');
    assert.equal(digest(Buffer.from(await response.arrayBuffer())), digest(readFileSync((process.env.CORPUS_HOST_PACKAGE || 'tmp/corpus-review-host/v1') + '/public/' + name)));
  }
  assert.equal((await fetch(base + '/research/corpus/api', { method: 'POST', headers: { authorization, origin: 'https://wrong.example' }, body: '{}' })).status, 403);
  assert.equal((await fetch(base)).status, 200);
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    mkdirSync('tmp/corpus-ui-checks', { recursive: true });
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 900 }, httpCredentials: { username: 'reviewer', password } });
      const errors = []; page.on('pageerror', e => errors.push(e.message));
      await page.goto(base + '/research/corpus/');
      await page.getByText('His supporters were very rowdy during the campaign.', { exact: true }).waitFor();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      assert.equal(await page.evaluate(() => {
        const top = document.querySelector('footer').getBoundingClientRect().top;
        return Array.from(document.querySelectorAll('textarea')).some(field => field.getBoundingClientRect().bottom > top);
      }), false);
      assert.deepEqual(errors, []);
      await page.screenshot({ path: `tmp/corpus-ui-checks/https-${width}.png`, fullPage: true });
      if (process.env.CORPUS_REFERENCE_REVIEW) {
        await page.goto(base + '/research/corpus/?collection=teacher');
        await page.getByText('Candidate annotations need review', { exact: true }).waitFor();
        assert.equal(await page.getByRole('radio').count(), 2);
        await page.getByText('Meaning and intent', { exact: true }).first().click();
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        await page.screenshot({ path: `tmp/corpus-ui-checks/https-annotations-${width}.png`, fullPage: true });
      }
      if (process.env.CORPUS_RESPONSE_REVIEW) {
        const expected = JSON.parse(readFileSync(process.env.CORPUS_RESPONSE_REVIEW, 'utf8'));
        const pendingOffset = expected.findIndex(row => !row.response.proposed);
        assert.ok(pendingOffset >= 0);
        await page.goto(base + '/research/corpus/?collection=responses&offset=0');
        await page.getByRole('heading', { name: 'Proposed Twi correction', exact: true }).waitFor();
        assert.equal(await page.getByRole('radio').count(), 2);
        assert.equal(await page.getByLabel('Correction for').inputValue(), 'conversation');
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        await page.screenshot({ path: `tmp/corpus-ui-checks/https-responses-${width}.png`, fullPage: true });
        await page.goto(base + '/research/corpus/?collection=responses&offset=' + pendingOffset);
        await page.getByRole('status').filter({ hasText: 'Correction pending' }).waitFor();
        assert.equal(await page.getByRole('radio').count(), 1);
        await page.reload();
        await page.getByRole('status').filter({ hasText: 'Correction pending' }).waitFor();
        assert.equal(await page.getByLabel('Collection').inputValue(), 'responses');
        assert.ok(page.url().includes('offset=' + pendingOffset));
        const corrected = expected.findIndex(row => row.response.corrected_english);
        await page.goto(base + '/research/corpus/?collection=responses&offset=' + corrected);
        await page.getByRole('heading', { name: 'Corrected English reference', exact: true }).waitFor();
        await page.getByText('Original English reference', { exact: true }).click();
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        await page.screenshot({ path: `tmp/corpus-ui-checks/https-source-correction-${width}.png`, fullPage: true });
        await page.getByRole('link', { name: 'Execution map', exact: true }).click();
        await page.getByRole('heading', { name: 'Corpus execution map', exact: true }).waitFor();
        await page.locator('#node-D summary').click();
        await page.locator('#node-E summary').click();
        assert.equal(await page.locator('#node-D details').getAttribute('open'), '');
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        await page.screenshot({ path: `tmp/corpus-ui-checks/https-progress-${width}.png`, fullPage: true });
      }
      assert.deepEqual(errors, []);
      await page.close();
    }
  } finally { await browser.close(); }
  console.log('HTTPS access, four real collections, asset checksums, CSRF, desktop/mobile and non-overlapping controls passed. No reviews written.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
