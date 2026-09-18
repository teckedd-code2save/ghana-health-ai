const { chromium, expect } = require('playwright/test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { mkdirSync, readFileSync, rmSync } = require('node:fs');
const path = require('node:path');

// Isolated fixtures and local review persistence; never connect to Postgres.
delete process.env.DATABASE_URL;
const { GET, POST } = require('../src/app/api/research/corpus-release/route.ts');
const release = `test-corpus-ui-${Date.now()}`;
const folder = path.resolve('tmp/corpus-releases', release);
const reviews = path.resolve('tmp/corpus-review', release);
mkdirSync(folder, { recursive: true });
execFileSync('python3', ['-c', `
import json, sqlite3, zlib, sys
db=sqlite3.connect(sys.argv[1]+'/ledger.sqlite3')
db.execute('CREATE TABLE records(n INTEGER PRIMARY KEY,id TEXT,payload BLOB,flags TEXT,split TEXT,disposition TEXT)')
db.execute('CREATE TABLE review_index(n INTEGER PRIMARY KEY,record_type TEXT,source TEXT,synthetic INTEGER,split TEXT,disposition TEXT)')
for i in range(2):
 row={'id':'test:'+str(i),'record_hash':str(i+1)*64,'text':'Medaase.' if i==0 else 'Maakye.', 'english':'Thank you.' if i==0 else 'Good morning.', 'source':'UI fixture, not training data','record_type':'sentence','language':'tw'}
 db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)',(i,row['id'],zlib.compress(json.dumps(row).encode()),'[]','train','accepted'))
 db.execute('INSERT INTO review_index VALUES(?,?,?,?,?,?)',(i,'sentence','fixture',0,'train','accepted'))
db.commit(); db.close()
`, folder]);

async function main() {
  const url = `http://localhost:3100/research/ase?release=${release}`;
  const apiUrl = `http://localhost:3100/api/research/corpus-release?release=${release}&collection=meaning&offset=0`;
  const sample = await (await GET(new Request(apiUrl))).json();
  assert.equal(sample.rows[0].original, 'Medaase.');
  const base = { release, collection: 'meaning', offset: 0, id: 'test:0', source_hash: '1'.repeat(64),
    selected: -1, correction: 'Thank you.', correction_field: 'meaning_english', issue: 'none', notes: 'Automated fixture only', decision: 'needs_second_review', request_id: crypto.randomUUID() };
  function request(body, origin = 'http://localhost:3100') {
    return new Request('http://localhost:3100/api/research/corpus-release', { method: 'POST', headers: { 'Content-Type': 'application/json', origin }, body: JSON.stringify(body) });
  }
  assert.equal((await POST(request(base, 'https://untrusted.example'))).status, 403);
  assert.equal((await POST(request({ ...base, source_hash: 'f'.repeat(64) }))).status, 409);
  assert.equal((await POST(request({ ...base, selected: 1 }))).status, 400);
  assert.equal((await POST(request(base))).status, 200);
  assert.equal((await POST(request(base))).status, 200);
  assert.equal(readFileSync(path.join(reviews, 'decisions.jsonl'), 'utf8').trim().split('\n').length, 1);
  assert.equal((await POST(request({ ...base, correction: 'Changed request with reused ID' }))).status, 409);
  const ledgerBefore = readFileSync(path.join(folder, 'ledger.sqlite3'));
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    mkdirSync('tmp/corpus-ui-checks', { recursive: true });
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      await page.route('**/api/research/corpus-release', async route => {
        const req = route.request();
        if (req.method() !== 'POST') return route.continue();
        const response = await POST(request(JSON.parse(req.postData())));
        await route.fulfill({ status: response.status, contentType: 'application/json', body: await response.text() });
      });
      await page.goto(url);
      await expect(page.getByText('Medaase.', { exact: true })).toBeVisible();
      await page.getByRole('textbox', { name: 'Correction', exact: true }).fill('Thank you very much.');
      await page.getByRole('combobox', { name: 'Issue', exact: true }).selectOption('naturalness');
      await page.getByRole('button', { name: 'Next sample', exact: true }).click();
      await expect(page.getByText('Maakye.', { exact: true })).toBeVisible();
      await page.reload();
      await expect(page.getByText('Maakye.', { exact: true })).toBeVisible();
      await page.getByRole('button', { name: 'Previous sample', exact: true }).click();
      await expect(page.getByRole('textbox', { name: 'Correction', exact: true })).toHaveValue('Thank you very much.');
      await expect(page.getByRole('combobox', { name: 'Issue', exact: true })).toHaveValue('naturalness');
      await page.getByRole('button', { name: 'Save correction', exact: true }).click();
      await expect(page.getByRole('status')).toHaveText('Saved on this device');
      await page.reload();
      await expect(page.getByRole('textbox', { name: 'Correction', exact: true })).toHaveValue('Thank you very much.');
      const dimensions = await page.evaluate(() => ({ content: document.documentElement.scrollWidth, viewport: innerWidth }));
      assert.ok(dimensions.content <= dimensions.viewport, JSON.stringify(dimensions));
      assert.deepEqual(errors, []);
      await page.screenshot({ path: `tmp/corpus-ui-checks/fixture-${viewport.width}.png`, fullPage: true });
      await page.close();
    }
  } finally { await browser.close(); }
  assert.deepEqual(readFileSync(path.join(folder, 'ledger.sqlite3')), ledgerBefore);
  console.log('Corpus review: desktop/mobile, refresh/navigation drafts, save/reload, immutable sources, stale hash rejection, CSRF and idempotency passed. No corpus reviews or database writes.');
}

main().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => {
  rmSync(folder, { recursive: true, force: true });
  rmSync(reviews, { recursive: true, force: true });
});
