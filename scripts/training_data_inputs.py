"""Incremental prepared batches and immutable input snapshots, not admission seals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile

POLICY = "training-input-snapshot-v1"
COLLECTIONS = {"language", "meaning", "responses", "actions", "english_retention"}
ARTIFACTS = {"batches.json", "records.jsonl", "split-ledger.jsonl", "report.json"}
SPLITS = {"train", "validation", "test"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = item
        return result

    def invalid_constant(value):
        raise ValueError("Non-finite JSON value: " + value)

    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid_constant)


def read(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, values):
    with path.open("w", encoding="utf-8") as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, parse_json(line)


def safe_path(folder, name):
    require(isinstance(name, str) and name and "\\" not in name and "\0" not in name,
            "Invalid artifact path")
    require(not Path(name).is_absolute() and ".." not in Path(name).parts
            and Path(name).as_posix() == name and name != ".", "Artifact escapes release")
    path = folder / name
    require(path.resolve().is_relative_to(folder.resolve()), "Artifact escapes release")
    for component in (path, *path.parents):
        if component == folder:
            break
        require(not component.is_symlink(), "Symlink artifact is not immutable")
    return path


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", value)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)


def check_batch(batch):
    require(isinstance(batch, dict) and type(batch.get("version")) is int and batch["version"] == 1,
            "Unknown batch version")
    require(valid_id(batch.get("batch_id")), "Invalid batch ID")
    require(batch.get("registration_is_semantic_approval") is False, "Registration cannot assert semantic approval")
    require(batch["collection"] in COLLECTIONS, "Unknown collection")
    root = Path(batch["release_path"])
    require(root.is_absolute() and root == root.resolve(), "Release path must be canonical and absolute")
    require(valid_hash(batch["release_manifest_sha256"]), "Invalid release manifest hash")
    manifest_path = safe_path(root, "manifest.json")
    require(sha(manifest_path) == batch["release_manifest_sha256"], "Changed release manifest")
    manifest = read(manifest_path)
    seal = batch["seal"]
    require(seal["name"] in {"READY.json", "SEALED.json"}, "Prepared source seal required")
    seal_path = safe_path(root, seal["name"])
    require(sha(seal_path) == seal["sha256"]
            and read(seal_path).get("manifest_sha256") == batch["release_manifest_sha256"], "Changed readiness seal")
    require(isinstance(batch["files"], dict) and batch["files"], "Explicit source files required")
    for name, checksum in batch["files"].items():
        path = safe_path(root, name)
        require(name.endswith(".jsonl") and valid_hash(checksum) and manifest["artifacts"].get(name) == checksum
                and sha(path) == checksum, "Unbound or changed source file: " + name)


def register(registry, batch_id, collection, release, expected_manifest, filenames, seal="READY.json"):
    require(valid_id(batch_id), "Invalid batch ID")
    require(seal in {"READY.json", "SEALED.json"}, "Prepared source seal required")
    require(filenames and len(filenames) == len(set(filenames)), "Missing or repeated source file")
    root = release.resolve()
    manifest = read(safe_path(root, "manifest.json"))
    for name in filenames:
        safe_path(root, name)
        require(name.endswith(".jsonl") and name in manifest["artifacts"], "Unbound source file: " + name)
    batch = {"version": 1, "batch_id": batch_id, "collection": collection, "release_path": str(root),
             "release_manifest_sha256": expected_manifest,
             "seal": {"name": seal, "sha256": sha(safe_path(root, seal))},
             "files": {name: manifest["artifacts"][name] for name in sorted(filenames)},
             "registration_is_semantic_approval": False}
    check_batch(batch)
    registry.mkdir(parents=True, exist_ok=True)
    path = safe_path(registry, batch_id + ".json")
    if path.exists():
        require(read(path) == batch, "Batch ID already binds different data; use a new version")
        return {"batch_id": batch_id, "state": "already_registered", "sha256": sha(path)}
    # Publish complete descriptors without overwriting a concurrent registration.
    with tempfile.NamedTemporaryFile(mode="w", dir=registry, delete=False, encoding="utf-8") as handle:
        temp = Path(handle.name)
        json.dump(batch, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    try:
        try:
            path.hardlink_to(temp)
        except FileExistsError:
            require(read(safe_path(registry, path.name)) == batch, "Concurrent conflicting registration")
    finally:
        temp.unlink()
    return {"batch_id": batch_id, "state": "registered", "sha256": sha(path)}


def row_identity(row, collection):
    require(isinstance(row, dict), "Input row must be an object")
    require(collection in COLLECTIONS, "Unknown collection")
    for field in ("id", "source_id", "source_hash", "group_id"):
        require(isinstance(row.get(field), str) and row[field].strip() and row[field] == row[field].strip(),
                "Missing or malformed source identity: " + field)
    require(valid_hash(row["source_hash"]), "Invalid source hash")
    require(row.get("split") in SPLITS, "Unassigned/protected records cannot become training inputs")
    require("training_eligible" not in row or row["training_eligible"] is True, "Source explicitly ineligible or invalid eligibility")
    require("protected" not in row or row["protected"] is False, "Protected records cannot become training inputs")
    for field in ("history_complete", "complete_history"):
        require(field not in row or row[field] is True, "Incomplete conversation")
    keys = {"source:" + row["source_id"], "group:" + row["group_id"]}
    for field in ("group_aliases", "inherited_group_ids", "related_tree_ids", "source_ids"):
        values = row.get(field, [])
        require(isinstance(values, list) and all(isinstance(v, str) and v.strip() and v == v.strip() for v in values), "Invalid aliases")
        keys.update(("source:" if field == "source_ids" else "group:") + value for value in values)
    for field, prefix in (("tree_id", "group:"), ("original_group_id", "group:"), ("source_group", "group:"),
                          ("family_id", "family:"), ("canonical_source_id", "source:")):
        if field in row:
            require(isinstance(row[field], str) and row[field].strip() and row[field] == row[field].strip(), "Invalid alias: " + field)
            keys.add(prefix + row[field])
    if collection == "language":
        require(isinstance(row.get("text"), str) and row["text"].strip(), "Empty language text")
        require(isinstance(row.get("language", "tw"), str) and row.get("language", "tw").strip(), "Invalid language")
        content = [row.get("language", "tw"), row["text"]]
    else:
        messages = row.get("messages")
        require(isinstance(messages, list) and messages, "Missing complete messages")
        require(all(isinstance(message, dict) for message in messages), "Invalid messages")
        start = int(messages[0].get("role") == "system")
        require(len(messages[start:]) > 0 and len(messages[start:]) % 2 == 0, "Incomplete conversation")
        for index, message in enumerate(messages):
            role = "system" if start and index == 0 else "user" if (index - start) % 2 == 0 else "assistant"
            require(message.get("role") == role and isinstance(message.get("content"), str)
                    and message["content"].strip(), "Invalid message roles/content")
        content = messages
    content_hash = digest(content)
    keys.add("content:" + collection + ":" + content_hash)
    return sorted(keys), content_hash


def verify_snapshot(folder):
    require(folder.is_dir() and not folder.is_symlink(), "Use a complete immutable snapshot directory")
    require({p.name for p in folder.iterdir()} == ARTIFACTS | {"manifest.json"}, "Incomplete or unexpected snapshot files")
    manifest = read(safe_path(folder, "manifest.json"))
    require(manifest["policy"] == POLICY and manifest["code_sha256"] == sha(__file__), "Changed snapshot implementation")
    require(manifest["training_authorized"] is False, "Input snapshot cannot authorize training")
    require(manifest["parent_manifest_sha256"] is None or valid_hash(manifest["parent_manifest_sha256"]), "Invalid parent binding")
    require(set(manifest["artifacts"]) == ARTIFACTS, "Incomplete snapshot")
    for name, checksum in manifest["artifacts"].items():
        require(sha(safe_path(folder, name)) == checksum, "Changed snapshot artifact")
    batches = read(folder / "batches.json")
    require(isinstance(batches, list) and batches, "Missing snapshot batches")
    by_id = {}
    for batch in batches:
        check_batch(batch)
        require(batch["batch_id"] not in by_id, "Repeated snapshot batch")
        by_id[batch["batch_id"]] = batch
    report = read(folder / "report.json")
    require(report.get("training_authorized") is False, "Snapshot report cannot authorize training")
    require(report["selected_batches"] == sorted(by_id), "Changed snapshot selection")
    pointers = set()
    for _, entry in rows(folder / "records.jsonl"):
        require(isinstance(entry, dict) and entry.get("batch_id") in by_id, "Unbound input row")
        batch = by_id[entry["batch_id"]]
        require(entry.get("file") in batch["files"] and entry.get("collection") == batch["collection"], "Unbound input file/collection")
        require(type(entry.get("line")) is int and entry["line"] > 0 and valid_hash(entry.get("row_sha256"))
                and entry.get("split") in SPLITS, "Invalid input row reference")
        pointer = (entry["batch_id"], entry["file"], entry["line"])
        require(pointer not in pointers, "Repeated input row reference")
        pointers.add(pointer)
    require(report["model_input_views"] == len(pointers), "Changed snapshot count")
    check_ledger([r for _, r in rows(folder / "split-ledger.jsonl")])
    return manifest


def check_ledger(ledger):
    # Union every link before checking splits, including witnesses dropped by children.
    roots, split_sets, source_hashes = {}, {}, {}

    def find(key):
        roots.setdefault(key, key)
        while roots[key] != key:
            roots[key] = roots[roots[key]]
            key = roots[key]
        return key

    for item in ledger:
        require(isinstance(item, dict) and item.get("split") in SPLITS, "Invalid split ledger")
        keys = item.get("keys")
        require(isinstance(keys, list) and keys and all(isinstance(k, str) and k for k in keys), "Incomplete split ledger")
        require(isinstance(item.get("source_id"), str) and "source:" + item["source_id"] in keys
                and valid_hash(item.get("source_hash")), "Invalid ledger source")
        require(source_hashes.setdefault(item["source_id"], item["source_hash"]) == item["source_hash"], "Source ID changed original hash")
        head = find(keys[0])
        for key in keys[1:]:
            roots[find(key)] = head
    for item in ledger:
        split_sets.setdefault(find(item["keys"][0]), set()).add(item["split"])
    require(all(len(v) == 1 for v in split_sets.values()), "Cross-split source/group/content conflict; preserve holdouts and repair batch upstream")


def publish_snapshot(staged, out):
    """Reserve the directory exclusively; manifest-last publication fails closed.

    Unlike rename(), mkdir() never replaces a concurrent empty directory. A crash
    can leave an incomplete directory, which must not be reused or overwritten.
    """
    out.mkdir()
    for name in sorted(ARTIFACTS) + ["manifest.json"]:
        (out / name).hardlink_to(staged / name)


def snapshot(registry, batch_ids, out, parent=None):
    require(not out.is_symlink(), "Snapshot directory cannot be a symlink")
    require(parent is None or parent.resolve() != out.resolve(), "Snapshot cannot be its own parent")
    require(batch_ids and len(batch_ids) == len(set(batch_ids)), "Explicit unique batch selection required")
    for key in batch_ids:
        require(valid_id(key), "Invalid batch ID")
    batches = [read(safe_path(registry, key + ".json")) for key in sorted(batch_ids)]
    require([b["batch_id"] for b in batches] == sorted(batch_ids), "Changed batch identity")
    ledger = [r for _, r in rows(parent / "split-ledger.jsonl")] if parent and verify_snapshot(parent) else []
    prior_batches = {b["batch_id"]: b for b in read(parent / "batches.json")} if parent else {}
    prior_ids = set(prior_batches)
    identities, seen_content, entries, duplicates = {}, {}, [], []
    source_hashes = {}
    for prior in ledger:
        source_hashes[prior["source_id"]] = prior["source_hash"]
    for batch in batches:
        check_batch(batch)
        require(batch["batch_id"] not in prior_batches or prior_batches[batch["batch_id"]] == batch,
                "Parent batch ID already binds different data")
        collection = batch["collection"]
        for name in batch["files"]:
            for line, row in rows(Path(batch["release_path"]) / name):
                keys, content = row_identity(row, collection)
                require(source_hashes.setdefault(row["source_id"], row["source_hash"]) == row["source_hash"], "Source ID changed original hash")
                ledger.append({"keys": keys, "split": row["split"], "source_id": row["source_id"], "source_hash": row["source_hash"]})
                version = digest(row)
                identity = (collection, row["id"])
                require(identities.setdefault(identity, version) == version, "Conflicting record versions; explicitly replace the old batch")
                entry = {"collection": collection, "id": row["id"], "split": row["split"], "source_id": row["source_id"],
                         "batch_id": batch["batch_id"], "file": name, "line": line, "row_sha256": version}
                canonical = (collection, content)
                if canonical in seen_content:
                    duplicates.append({**entry, "representative": seen_content[canonical]})
                else:
                    seen_content[canonical] = {"batch_id": batch["batch_id"], "id": row["id"]}
                    entries.append(entry)
    check_ledger(ledger)
    ledger = list({digest(r): r for r in ledger}.values())
    counts = Counter((r["collection"], r["split"]) for r in entries)
    report = {"selected_batches": sorted(batch_ids), "added_batches": sorted(set(batch_ids) - prior_ids),
              "removed_batches": sorted(prior_ids - set(batch_ids)), "model_input_views": len(entries),
              "counts": [{"collection": c, "split": s, "views": count} for (c, s), count in sorted(counts.items())],
              "duplicate_views_not_repeated": duplicates, "training_authorized": False,
              "limits": "Prepared-source composition only; seals are hash bindings, not independently authenticated approvals. No tokenization, semantic admission or training authorization. Exact within-collection content deduplication only; no normalization or near-duplicate scan. Split protection covers declared identities in selected batches and the cumulative parent ledger, not unseen sources or unrelated snapshot branches. Alternating message roles do not prove source-history completeness. Existing semantic, licensing, fresh-review, near-duplicate/holdout and tokenizer gates still apply. Bidirectional translation is two model-input views, not two independent source pairs. Original release files must remain available and unchanged; this snapshot stores references, not raw copies."}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".input-snapshot-") as temp:
        staged = Path(temp) / "snapshot"
        staged.mkdir()
        save(staged / "batches.json", batches)
        write_rows(staged / "records.jsonl", entries)
        write_rows(staged / "split-ledger.jsonl", ledger)
        save(staged / "report.json", report)
        save(staged / "manifest.json", {"policy": POLICY, "code_sha256": sha(__file__),
            "parent_manifest_sha256": sha(parent / "manifest.json") if parent else None,
            "training_authorized": False, "artifacts": {p.name: sha(p) for p in sorted(staged.iterdir())}})
        verify_snapshot(staged)
        try:
            publish_snapshot(staged, out)
        except FileExistsError:
            existing = verify_snapshot(out)
            require(existing == read(staged / "manifest.json"), "Snapshot already binds different inputs; use a new directory")
    return report


def iter_snapshot_records(folder, collection=None, split=None):
    """Feed original rows to the existing task-specific prep/tokenization gates."""
    require(collection is None or collection in COLLECTIONS, "Unknown collection")
    require(split is None or split in SPLITS, "Unknown split")
    verify_snapshot(folder)
    batches = {b["batch_id"]: b for b in read(folder / "batches.json")}
    selected = {}
    for _, entry in rows(folder / "records.jsonl"):
        if (collection and entry["collection"] != collection) or (split and entry["split"] != split):
            continue
        selected.setdefault((entry["batch_id"], entry["file"]), {})[entry["line"]] = entry
    for (batch_id, filename), line_map in sorted(selected.items()):
        found = set()
        for number, row in rows(safe_path(Path(batches[batch_id]["release_path"]), filename)):
            if number in line_map:
                entry = line_map[number]
                require(digest(row) == entry["row_sha256"] and row["id"] == entry["id"]
                        and row["split"] == entry["split"] and row["source_id"] == entry["source_id"], "Changed input row")
                row_identity(row, entry["collection"])
                found.add(number)
                yield {"collection": entry["collection"], "batch_id": batch_id, "row": row}
        require(found == set(line_map), "Missing selected input rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    add = sub.add_parser("register")
    add.add_argument("--registry", type=Path, required=True)
    add.add_argument("--batch-id", required=True)
    add.add_argument("--collection", choices=sorted(COLLECTIONS), required=True)
    add.add_argument("--release", type=Path, required=True)
    add.add_argument("--manifest-sha256", required=True)
    add.add_argument("--file", action="append", required=True)
    add.add_argument("--seal", choices=("READY.json", "SEALED.json"), default="READY.json")
    freeze = sub.add_parser("snapshot")
    freeze.add_argument("--registry", type=Path, required=True)
    freeze.add_argument("--batch", action="append", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--parent", type=Path)
    check = sub.add_parser("verify")
    check.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "register":
        value = register(args.registry, args.batch_id, args.collection, args.release, args.manifest_sha256, args.file, args.seal)
    elif args.action == "snapshot":
        value = snapshot(args.registry, args.batch, args.out, args.parent)
    else:
        value = {"manifest": verify_snapshot(args.snapshot), "verified_rows": sum(1 for _ in iter_snapshot_records(args.snapshot))}
    print(json.dumps(value, indent=2))
