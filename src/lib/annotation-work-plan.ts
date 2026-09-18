import fs from "node:fs/promises";

export async function annotationWorkPlan(input: {
  file: string; cacheDirectory: string; signature: string;
  selectedIds: string[]; pendingIds: string[]; chunkSize: number; force?: boolean;
}) {
  const selected = new Set(input.selectedIds);
  try {
    const saved = JSON.parse(await fs.readFile(input.file, "utf8")) as { signature: string; chunks: string[][] };
    if (!input.force) {
      const ids = saved.chunks.flat();
      if (saved.signature !== input.signature || new Set(ids).size !== ids.length ||
        ids.some((id) => !selected.has(id)) || input.pendingIds.some((id) => !ids.includes(id))) {
        throw new Error("Annotation work plan does not match the source selection; use a new output.");
      }
      return saved.chunks;
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const pending = new Set(input.pendingIds);
  const index = new Map(input.selectedIds.map((id, i) => [id, i]));
  const groups = new Map<string, string[]>();
  let files: string[] = [];
  try { files = await fs.readdir(input.cacheDirectory); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  for (const file of files.filter((file) => file.endsWith(".json") && !file.includes(".attempt-"))) {
    const checkpoint = JSON.parse(await fs.readFile(`${input.cacheDirectory}/${file}`, "utf8"));
    const ids: string[] = (checkpoint.payload?.rows ?? []).map((row: { row_id?: string }) => row.row_id);
    if (!ids.length || ids.length > input.chunkSize || ids.some((id) => !selected.has(id)) || new Set(ids).size !== ids.length) continue;
    if (!ids.some((id) => pending.has(id))) continue;
    ids.sort((a, b) => index.get(a)! - index.get(b)!);
    groups.set(JSON.stringify(ids), ids);
  }
  const used = new Set<string>();
  const chunks: string[][] = [];
  // Preserve paid multi-row requests when recovering an older run without a plan.
  for (const ids of [...groups.values()].sort((a, b) => b.length - a.length || a[0].localeCompare(b[0]))) {
    if (ids.some((id) => used.has(id))) continue;
    chunks.push(ids);
    ids.forEach((id) => used.add(id));
  }
  const remaining = input.pendingIds.filter((id) => !used.has(id));
  for (let i = 0; i < remaining.length; i += input.chunkSize) chunks.push(remaining.slice(i, i + input.chunkSize));
  const temporary = `${input.file}.${process.pid}.tmp`;
  await fs.writeFile(temporary, JSON.stringify({ schema_version: 1, signature: input.signature, chunks }, null, 2) + "\n");
  await fs.rename(temporary, input.file);
  return chunks;
}
