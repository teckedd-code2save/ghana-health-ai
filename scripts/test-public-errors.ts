import assert from "node:assert/strict";
import { test } from "node:test";
import { publicFailure, unclearRecording } from "../src/lib/public-errors";
test("database capacity and recovery are service failures, never transcription advice", () => {
  for (const error of [new Error('No space left on device'), new Error('the database system is in recovery mode'), new Error('Database adapter error', { cause: { code: '53100' } })]) {
    const failure = publicFailure(error, 'voice');
    assert.equal(failure.status, 503);
    assert.match(failure.message, /cannot save/);
    assert.doesNotMatch(failure.message, /closer|Postgres|53100|No space/);
  }
  assert.doesNotMatch(publicFailure(new Error('unclassified'), 'voice').message, /microphone|closer/);
  assert.equal(unclearRecording.status, 422);
});
