import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { annotationWorkPlan } from "../src/lib/annotation-work-plan";

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "annotation-plan-test-"));
  try {
    const cacheDirectory = path.join(root, "cache");
    await fs.mkdir(cacheDirectory);
    await fs.writeFile(path.join(cacheDirectory, "cached.json"), JSON.stringify({ payload: { rows: [{ row_id: "a" }, { row_id: "b" }] } }));
    const input = { file: path.join(root, "plan.json"), cacheDirectory, signature: "test", selectedIds: ["a", "b", "c"], pendingIds: ["b", "c"], chunkSize: 2 };
    const first = await annotationWorkPlan(input);
    assert.deepEqual(first, [["a", "b"], ["c"]], "Preserve the paid request containing both the completed and incomplete row");
    assert.deepEqual(await annotationWorkPlan({ ...input, pendingIds: ["c"] }), first, "Resume must not regroup requests after completion");
    await assert.rejects(() => annotationWorkPlan({ ...input, signature: "changed" }), /does not match/);
    await assert.rejects(() => annotationWorkPlan({ ...input, selectedIds: ["a", "b", "c", "d"], pendingIds: ["d"] }), /does not match/);
    const fresh = await annotationWorkPlan({ ...input, file: path.join(root, "fresh.json"), cacheDirectory: path.join(root, "absent"), pendingIds: ["a", "b", "c"] });
    assert.deepEqual(fresh, [["a", "b"], ["c"]]);
    console.log("Annotation plan: paid batch recovery, stable resumes, source mismatch rejection and fresh batching passed.");
  } finally { await fs.rm(root, { recursive: true, force: true }); }
}
void main().catch((error) => { console.error(error); process.exitCode = 1; });
