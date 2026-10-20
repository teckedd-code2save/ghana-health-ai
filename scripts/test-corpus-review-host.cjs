/* eslint-disable @typescript-eslint/no-require-imports -- Node CommonJS fixture harness. */
const assert = require('node:assert/strict');
const { spawn, execFileSync } = require('node:child_process');
const { mkdtempSync, mkdirSync, cpSync, existsSync, readFileSync, writeFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { createHash, randomBytes, randomUUID } = require('node:crypto');
const { chromium } = require('playwright');

async function main() {
  const folder = mkdtempSync(path.join(tmpdir(), 'corpus-host-test-'));
  const packaged = path.resolve(process.env.CORPUS_HOST_PACKAGE || 'tmp/corpus-review-host/v1');
  const release = 'twi-stage-v1-20260911';
  const data = path.join(folder, 'tmp/corpus-releases', release);
  mkdirSync(data, { recursive: true });
  cpSync(path.join(packaged, 'server.cjs'), path.join(folder, 'server.cjs'));
  cpSync(path.join(packaged, 'public'), path.join(folder, 'public'), { recursive: true });
  cpSync(path.join(packaged, 'scripts'), path.join(folder, 'scripts'), { recursive: true });
  if (!existsSync(path.join(folder, 'public/progress.html'))) {
    writeFileSync(path.join(folder, 'public/progress.html'), '<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/research/corpus/progress.css"></head><body><main><h1>Corpus execution map</h1><ol><li id="node-D"><details><summary><b>D</b><span>Fixture corrections</span><em>In progress</em></summary><p>Disposable test fixture; no corpus records.</p></details></li></ol></main></body></html>');
  }
  execFileSync('python3', ['-c', `
import json,sqlite3,zlib,sys
d=sqlite3.connect(sys.argv[1])
d.execute('CREATE TABLE records(n INTEGER PRIMARY KEY,id TEXT,payload BLOB,flags TEXT)')
d.execute('CREATE TABLE review_index(n INTEGER PRIMARY KEY,record_type TEXT,source TEXT,synthetic INTEGER,split TEXT,disposition TEXT)')
r={'id':'test:0','record_hash':'1'*64,'text':'Medaase.','english':'Thank you.','source':'Test fixture','language':'tw','record_type':'sentence'}
d.execute('INSERT INTO records VALUES(0,?,?,?)',(r['id'],zlib.compress(json.dumps(r).encode()),'[]'))
d.execute("INSERT INTO review_index VALUES(0,'sentence','fixture',0,'train','accepted')")
d.commit();d.close()
`, path.join(data, 'ledger.sqlite3')]);
  const annotations = path.join(folder, 'tmp/corpus-reference-review', release);
  mkdirSync(annotations, { recursive: true });
  execFileSync('python3', ['-c', `
import sqlite3,json,sys
d=sqlite3.connect(sys.argv[1]);d.execute('CREATE TABLE annotations(source_id TEXT PRIMARY KEY,source_hash TEXT,payload TEXT)')
a={'id':'3'*64,'reference_english':'Thank you.','analysis':{'record_function':'narrative','conversational_intent':None,'entities':[],'meaning_features':{},'ambiguity':[]}}
d.execute('INSERT INTO annotations VALUES(?,?,?)',('test:0','1'*64,json.dumps(a)));d.commit();d.close()
`, path.join(annotations, 'annotations.sqlite3')]);
  writeFileSync(path.join(annotations, 'qualification-review.json'), JSON.stringify([{
    id: 'test:0', source_hash: '1'.repeat(64), original_tw: 'Medaase.', reference_en: 'Thank you.',
    source: 'Test fixture', language: 'tw', split: 'validation', training_eligible: false,
    reference_analysis: { id: '5'.repeat(64), analysis: null },
    annotation_status: 'Candidate annotations need review',
    alternatives: [{ model: 'Candidate A', text: 'narrative', fields: { intent: null } },
      { model: 'Candidate B', text: 'narrative', fields: { intent: null } }],
  }]));
  const responses = path.join(folder, 'tmp/corpus-response-review', release);
  mkdirSync(responses, { recursive: true });
  const responseRow = { id: 'response:0', source_hash: '6'.repeat(64), proposal_set_id: '7'.repeat(64),
    source: 'Response test fixture', language: 'tw', training_eligible: false,
    original: 'Fixture request', annotation_status: 'Correction proposed; not approved',
    response: { original: { instruction: 'Original request', input: 'Original context', output: 'Original answer' },
      english: { instruction: 'English request', input: 'English context', output: 'Complete English answer' },
      corrected_english: { instruction: 'English request', input: 'English context', output: 'Corrected English fixture answer' },
      proposed: { instruction: 'Corrected request', input: 'Corrected context', output: 'Complete corrected answer' },
      remaining_review: ['Fixture naturalness review'], issues: ['fixture issue'] },
    alternatives: [{ model: 'Original Twi', text: '' }, { model: 'Proposed correction', text: '' }] };
  const writeResponses = () => {
    const text = JSON.stringify([responseRow]);
    writeFileSync(path.join(responses, 'queue.json'), text);
    writeFileSync(path.join(responses, 'manifest.json'), JSON.stringify({ sources: 1,
      queue_sha256: createHash('sha256').update(text).digest('hex') }));
  };
  writeResponses();
  const password = randomBytes(24).toString('hex');
  const origin = 'https://review.example';
  const base = 'http://127.0.0.1:13122';
  const authorization = 'Basic ' + Buffer.from('reviewer:' + password).toString('base64');
  const child = spawn(process.execPath, ['server.cjs'], { cwd: folder, env: {
    PATH: process.env.PATH, NODE_ENV: 'production', CORPUS_RELEASE: release, PORT: '13122', REVIEW_ORIGIN: origin, REVIEW_PASSWORD: password,
  }, stdio: ['ignore', 'pipe', 'pipe'] });
  try {
    await new Promise((resolve, reject) => { child.stdout.once('data', resolve); child.once('exit', code => reject(Error('Server exited ' + code))); });
    assert.equal((await fetch(base + '/research/corpus/')).status, 401);
    assert.equal((await fetch(base + '/research/corpus/app.js')).status, 401);
    assert.equal((await fetch(base + '/research/corpus/progress')).status, 401);
    assert.equal((await fetch(base + '/research/corpus/progress.css')).status, 401);
    assert.equal((await fetch(base + '/research/corpus/api?release=' + release + '&collection=meaning&offset=0')).status, 401);
    const api = base + '/research/corpus/api';
    const info = await (await fetch(api + '?release=' + release + '&collection=meaning&offset=0', { headers: { authorization } })).json();
    assert.equal(info.ready, true, JSON.stringify(info));
    assert.equal(info.rows[0].original, 'Medaase.');
    assert.equal(info.rows[0].reference_analysis.id, '3'.repeat(64));
    const payload = { release, collection: 'meaning', offset: 0, id: 'test:0', source_hash: '1'.repeat(64), selected: -1,
      correction: 'Thank you.', correction_field: 'meaning_english', issue: 'none', notes: 'Automated fixture', decision: 'needs_second_review', request_id: randomUUID() };
    const post = (value, source = origin) => fetch(api, { method: 'POST', headers: { authorization, origin: source, 'Content-Type': 'application/json' }, body: JSON.stringify(value) });
    assert.equal((await post(payload, 'https://wrong.example')).status, 403);
    assert.equal((await post({ ...payload, release: 'different' })).status, 404);
    assert.equal((await post({ ...payload, source_hash: '2'.repeat(64) })).status, 409);
    assert.equal((await post({ ...payload, reference_analysis_id: '4'.repeat(64) })).status, 409);
    payload.reference_analysis_id = '3'.repeat(64);
    assert.equal((await post(payload)).status, 200);
    assert.equal((await post(payload)).status, 200);
    assert.equal(readFileSync(path.join(folder, 'tmp/corpus-review', release, 'decisions.jsonl'), 'utf8').trim().split('\n').length, 1);
    const candidate = await (await fetch(api + '?release=' + release + '&collection=teacher&offset=0', { headers: { authorization } })).json();
    assert.equal(candidate.rows[0].source_hash, payload.source_hash);
    assert.equal(candidate.rows[0].training_eligible, false);
    assert.equal(candidate.rows[0].alternatives.length, 2);
    assert.equal((await post({ ...payload, collection: 'teacher', request_id: randomUUID() })).status, 409);
    const responsePayload = { ...payload, collection: 'responses', id: responseRow.id, source_hash: responseRow.source_hash,
      reference_analysis_id: undefined, proposal_set_id: responseRow.proposal_set_id, decision: 'reviewed',
      correction: '', correction_field: 'conversation', selected: 1, request_id: randomUUID() };
    assert.equal((await post({ ...responsePayload, proposal_set_id: undefined })).status, 409);
    assert.equal((await post({ ...responsePayload, proposal_set_id: '8'.repeat(64) })).status, 409);
    assert.equal((await post({ ...responsePayload, selected: -1 })).status, 400);
    assert.equal((await post(responsePayload)).status, 200);
    assert.equal((await post(responsePayload)).status, 200);
    const responseJournal = path.join(folder, 'tmp/corpus-review', release + '-responses', 'decisions.jsonl');
    assert.equal(readFileSync(responseJournal, 'utf8').trim().split('\n').length, 1);
    assert.equal(readFileSync(path.join(folder, 'tmp/corpus-review', release, 'decisions.jsonl'), 'utf8').trim().split('\n').length, 1);
    const browser = await chromium.launch({ channel: 'chrome', headless: true });
    try {
      for (const width of [1440, 390]) {
        const page = await browser.newPage({ viewport: { width, height: 900 }, httpCredentials: { username: 'reviewer', password } });
        // This HTTP-only fixture emulates the configured HTTPS reverse proxy.
        // The API assertions above still test rejection of a foreign Origin.
        await page.route('**/research/corpus/api', async route => {
          if (route.request().method() !== 'POST') return route.continue();
          const response = await route.fetch({ headers: { ...route.request().headers(), authorization, origin } });
          await route.fulfill({ response });
        });
        const errors = []; page.on('pageerror', e => errors.push(e.message));
        await page.goto(base + '/research/corpus/');
        await page.getByText('Medaase.', { exact: true }).waitFor();
        await page.getByRole('heading', { name: 'English-reference analysis' }).waitFor();
        assert.equal(await page.getByRole('textbox', { name: 'Correction', exact: true }).inputValue(), 'Thank you.');
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        const overlap = await page.evaluate(() => {
          const footer = document.querySelector('footer').getBoundingClientRect();
          return Array.from(document.querySelectorAll('textarea')).some(field => field.getBoundingClientRect().bottom > footer.top);
        });
        assert.equal(overlap, false, 'Footer overlaps correction fields');
        assert.deepEqual(errors, []);
        await page.goto(base + '/research/corpus/?collection=teacher');
        await page.getByText('Candidate A', { exact: true }).waitFor();
        assert.equal(await page.getByLabel('Collection').inputValue(), 'teacher');
        assert.equal(await page.getByRole('radio').count(), 2);
        assert.equal(await page.getByRole('radio').first().isChecked(), false);
        await page.getByRole('radio').first().check();
        const savedResponse = page.waitForResponse(response => response.request().method() === 'POST' && response.url().endsWith('/research/corpus/api'));
        await page.getByRole('button', { name: 'Save correction', exact: true }).click();
        const saved = await savedResponse;
        assert.equal(saved.status(), 200, await saved.text());
        await page.getByRole('status').filter({ hasText: /^Saved$/ }).waitFor();
        const decisions = readFileSync(path.join(folder, 'tmp/corpus-review', release, 'decisions.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
        assert.equal(decisions.at(-1).reference_analysis_id, '5'.repeat(64));
        assert.equal(decisions.at(-1).collection, 'teacher');
        // Remove only this page's fixture draft/review before the next viewport.
        writeFileSync(path.join(folder, 'tmp/corpus-review', release, 'decisions.jsonl'), JSON.stringify(decisions[0]) + '\n');
        await page.goto(base + '/research/corpus/?collection=responses');
        await page.getByText('Complete corrected answer', { exact: true }).waitFor();
        await page.getByRole('heading', { name: 'Corrected English reference', exact: true }).waitFor();
        await page.getByText('Corrected English fixture answer', { exact: true }).waitFor();
        await page.getByText('Original English reference', { exact: true }).click();
        await page.getByText('Complete English answer', { exact: true }).waitFor();
        await page.getByText('Original English reference', { exact: true }).click();
        assert.equal(await page.getByLabel('Correction for').inputValue(), 'conversation');
        assert.equal(await page.getByRole('radio', { name: 'Proposed correction', exact: true }).isChecked(), true);
        await page.getByRole('textbox', { name: 'Correction', exact: true }).fill('Full fixture correction');
        await page.reload();
        await page.getByText('Complete corrected answer', { exact: true }).waitFor();
        assert.equal(await page.getByRole('textbox', { name: 'Correction', exact: true }).inputValue(), 'Full fixture correction');
        const saveResponse = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/research/corpus/api'));
        await page.getByRole('button', { name: 'Save correction', exact: true }).click();
        assert.equal((await saveResponse).status(), 200);
        assert.equal(JSON.parse(readFileSync(responseJournal, 'utf8').trim().split('\n').at(-1)).proposal_set_id, '7'.repeat(64));
        responseRow.proposal_set_id = '9'.repeat(64);
        writeResponses();
        await page.reload();
        await page.getByText('Complete corrected answer', { exact: true }).waitFor();
        assert.equal(await page.getByRole('radio', { name: 'Proposed correction', exact: true }).isChecked(), false);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        mkdirSync('tmp/corpus-ui-checks', { recursive: true });
        await page.screenshot({ path: `tmp/corpus-ui-checks/response-fixture-${width}.png`, fullPage: true });
        await page.getByRole('link', { name: 'Execution map', exact: true }).click();
        await page.getByRole('heading', { name: 'Corpus execution map', exact: true }).waitFor();
        await page.locator('#node-D summary').click();
        assert.equal(await page.locator('#node-D details').getAttribute('open'), '');
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        await page.screenshot({ path: `tmp/corpus-ui-checks/execution-map-${width}.png`, fullPage: true });
        assert.deepEqual(errors, []);
        responseRow.proposal_set_id = '7'.repeat(64);
        writeResponses();
        writeFileSync(responseJournal, JSON.stringify(responsePayload) + '\n');
        await page.close();
      }
    } finally { await browser.close(); }
    console.log('Standalone HTTPS gateway: authentication, isolation, CSRF, source hashes, durable idempotent saves and responsive UI passed. Fixtures only.');
  } finally {
    child.kill('SIGTERM');
    await new Promise(resolve => child.once('exit', resolve));
    rmSync(folder, { recursive: true, force: true });
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
