#!/usr/bin/env python3
"""Immutable private R3 research snapshots. Offline freeze; explicit upload only."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

import run_understanding_r3 as runner
from run_understanding_r2 import read, require, safe_file, save, sha, VOLUME

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "/research-training-r3-20261106"
SCHEMA = "understanding-r3-durability-v1"
CODE = (
    "scripts/checkpoint_understanding_r3.py", "scripts/test_checkpoint_understanding_r3.py",
    "scripts/run_understanding_r3.py", "scripts/test_run_understanding_r3.py",
    "modal/train/train_understanding_r3.py", "scripts/prepare_understanding_r3.py",
    "scripts/test_prepare_understanding_r3.py", "scripts/repair_r3_conversations.py",
    "scripts/admit_understanding_r3_repairs.py", "scripts/test_admit_understanding_r3_repairs.py",
    "scripts/test_repair_r3_conversations.py", "scripts/prepare_understanding_r3_evaluation.py",
    "scripts/test_prepare_understanding_r3_evaluation.py", "scripts/run_understanding_r3_evaluation.py",
    "modal/train/evaluate_understanding_r3.py", "scripts/r3_inference.py",
    "scripts/prepare_understanding_r3_inference.py", "scripts/test_prepare_understanding_r3_inference.py",
    "scripts/understanding_r3_demo.py", "scripts/test_understanding_r3_demo.py",
    "modal/research_understanding_r3_demo.py", "scripts/r2_inference.py",
    "scripts/understanding_r2_demo.py", "scripts/understanding_r2_workbench.py",
    "scripts/r2_comparison.py", "modal/research_understanding_r2_demo.py",
    "docs/TWI_UNDERSTANDING_R3_20261106.md", "tmp/r3-runner-contract.md",
    "tmp/r3-repair-contract.md", "tmp/r3-evaluation-contract.md", "tmp/r3-ui-contract.md",
) + runner.REUSED
REGISTRATION = ("data/research/r3-repair-plan.handoff.v1.json", "data/research/r3-repair-parent-review.v1.json")
PACK_NAMES = {
    "manifest.json", "PREPARED.json", "rows.jsonl", "source-witnesses.jsonl", "stages.json",
    "evaluation-manifest.json", "overlap-audit.json", "held-content.jsonl", "cases.jsonl",
    "loss-items.jsonl", "protocol.json", "source-manifests.json", "admission-manifests.json",
    "selection-ledger.jsonl", "selection-report.json", "excluded.jsonl", "duplicate-prefixes.jsonl",
    "run-selection.json",
}
RECEIPTS = {
    "execution.json", "status.json", "loading.json", "smoke.json", "stages.json", "adapters.json",
    "resume.json", "restore-proof.json", "consumption.json", "launch.json", "receipt.json",
    "smoke.call.json", "train.call.json", "smoke.launch.json", "train.launch.json",
    "smoke.receipt.json", "train.receipt.json", "consumption-smoke-uninterrupted.json",
    "consumption-smoke-interrupted.json", "consumption-smoke-resumed.json",
}
REGISTERED_NAMES = {
    "README.md", "DATASET_CARD.md", "dataset-card.md", "manifest.json", "READY.json", "PREPARED.json",
    "sources.jsonl", "versions.jsonl", "review-candidates.jsonl", "admitted.jsonl", "unresolved.jsonl",
    "r3-mapping.jsonl", "request-costs.jsonl", "report.json", "lineage.json", "records.jsonl",
    "conversations.jsonl", "rejected.jsonl", "admission.json",
}
DENIED = re.compile(r"(?i)(?:sqlite|\.db(?:$|[-.])|observed-feedback-proofs|owner[-_]?reviews?|"
                    r"freeform|audition|email|attachment|\.env(?:$|\.)|(?:^|/)budget(?:[./_-]|$))")
SECRET = re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|\b(?:hf_|ghp_|github_pat_|"
                    rb"sk-(?:proj-|svcacct-)?|xox[baprs]-)[A-Za-z0-9_-]{20,}|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
SECRET_KEYS = {"api_key", "apikey", "access_token", "refresh_token", "secret_key", "client_secret",
               "authorization", "password", "token_secret", "modal_token_secret"}


def safe_name(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and name and not path.is_absolute() and ".." not in path.parts
            and "\\" not in name and str(path) == name and not DENIED.search(name), "Forbidden archive path: " + str(name))
    return name


def pack_allowed(name):
    safe_name(name)
    return (name in PACK_NAMES
            or re.fullmatch(r"tokenizer/(?:tokenizer\.json|chat_template\.jinja|config\.json|generation_config\.json)", name)
            or re.fullmatch(r"datasets/[a-z][a-z0-9_-]*/(?:manifest\.json|records\.jsonl|README\.md)", name))


def registered_allowed(name):
    safe_name(name)
    return (name in REGISTERED_NAMES
            or re.fullmatch(r"(?:requests|responses)/[a-f0-9]{64}\.json", name)
            or re.fullmatch(r"evidence/(?:batches|agent-reviews)/[A-Za-z0-9_.-]+\.json", name))


def receipt_allowed(name):
    safe_name(name)
    path = PurePosixPath(name)
    return path.name in RECEIPTS and (len(path.parts) == 1 or
        (len(path.parts) == 2 and re.fullmatch(r"(?:smoke|train|evidence|progress|collected)(?:[-_][A-Za-z0-9_-]+)*", path.parts[0])))


def member_allowed(name):
    safe_name(name)
    if name == "manifest.json":
        return True
    parts = name.split("/")
    group, relative = parts[0], "/".join(parts[1:])
    if group == "code":
        return relative in CODE
    if group == "frozen":
        return relative in runner.FROZEN
    if group == "registration":
        return relative in REGISTRATION
    if group == "pack":
        return pack_allowed(relative)
    if len(parts) < 3 or not parts[1].isdigit():
        return False
    relative = "/".join(parts[2:])
    if group == "registered":
        return registered_allowed(relative)
    if group == "receipts":
        return receipt_allowed(relative)
    if group == "executions":
        if relative == "execution.json":
            return True
        for prefix, names in (("r3/", runner.CODE), ("reused/", runner.REUSED), ("frozen/", runner.FROZEN)):
            if relative.startswith(prefix):
                return relative[len(prefix):] in names
        return relative.startswith("inputs/") and pack_allowed(relative.removeprefix("inputs/"))
    return False


def screen(name, raw):
    safe_name(name)
    require(not raw.startswith(b"SQLite format 3\x00") and not SECRET.search(raw), "Potential credential/database in " + name)
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                require(not (key.lower() in SECRET_KEYS and isinstance(item, str) and item.strip()), "Credential field in " + name)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    if name.endswith((".json", ".jsonl")):
        for item in raw.splitlines() if name.endswith(".jsonl") else [raw]:
            if item.strip():
                try:
                    value = json.loads(item)
                except (ValueError, UnicodeDecodeError):
                    continue  # Incomplete synthetic HTTP responses still receive byte screening.
                walk(value)


def freeze(pack, out, version, source, *, registered=(), executions=(), runs=(), phase="pre_results"):
    """Freeze explicit research inputs, without reading any live store or ledger."""
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", version), "Use a safe unique version label")
    require(phase in {"pre_results", "prelaunch", "postrun"}, "Unknown snapshot phase")
    if phase == "prelaunch":
        phase = "pre_results"  # Compatibility alias; a launch may already have occurred.
    pack, out, source = Path(pack).resolve(strict=True), Path(out).absolute(), Path(source).resolve(strict=True)
    require(not out.exists() and not out.is_symlink(), "Use a fresh snapshot directory")
    roots = [pack, source, ROOT] + [Path(p).resolve(strict=True) for p in (*registered, *executions, *runs)]
    require(not any(out.resolve() == root or out.resolve().is_relative_to(root) for root in [pack, *roots[3:]]), "Snapshot must be outside selected artifact roots")
    paths, omitted, provenance = {}, [], {"registered": [], "executions": [], "run_receipts": []}
    def add(prefix, root, name):
        key = safe_name(prefix + "/" + name)
        require(key not in paths, "Duplicate selected member")
        paths[key] = safe_file(root, name)
    for name in CODE:
        if (ROOT / name).exists() or (ROOT / name).is_symlink():
            add("code", ROOT, name)
    for name, checksum in runner.FROZEN.items():
        require(sha(safe_file(source, name)) == checksum, "Frozen dependency changed: " + name)
        add("frozen", source, name)
    manifest, _, _ = runner.verify_pack(pack)
    for name in sorted(set(manifest["artifacts"]) | {"manifest.json", "PREPARED.json"}):
        require(pack_allowed(name), "Pack contains non-allowlisted input: " + name)
        add("pack", pack, name)
    provenance["pack_manifest_sha256"] = sha(pack / "manifest.json")
    for index, folder in enumerate(registered):
        folder = Path(folder).resolve(strict=True)
        value = read(safe_file(folder, "manifest.json"))
        require(value.get("dataset_id") and value.get("dataset_version"), "Only registered versioned datasets may be selected")
        require(value.get("schema") == "r3-conversation-repair-dataset-v1", "Unknown registered synthetic dataset schema")
        files = value.get("files", value.get("artifacts"))
        require(isinstance(files, dict) and files, "Registered dataset needs a checksum inventory")
        selected = {"manifest.json"}
        for name, proof in files.items():
            try:
                allowed = registered_allowed(name)
            except ValueError:
                allowed = False
            if not allowed:
                omitted.append({"scope": "registered/" + str(index), "path": name, "reason": "not_allowlisted"})
                continue
            path = safe_file(folder, name)
            require(sha(path) == (proof["sha256"] if isinstance(proof, dict) else proof), "Registered artifact changed: " + name)
            if isinstance(proof, dict) and "bytes" in proof:
                require(path.stat().st_size == proof["bytes"], "Registered artifact size changed")
            selected.add(name)
        for name in sorted(selected):
            add("registered/" + str(index), folder, name)
        provenance["registered"].append({"dataset_id": value["dataset_id"], "dataset_version": value["dataset_version"],
                                         "manifest_sha256": sha(folder / "manifest.json"), "selected_files": len(selected)})
    registered_hashes = {r["manifest_sha256"] for r in provenance["registered"]}
    for name in REGISTRATION:
        if registered_hashes and (ROOT / name).is_file():
            value = read(safe_file(ROOT, name))
            field = "dataset_manifest_sha256" if "parent-review" in name else "manifest_sha256"
            require(value[field] in registered_hashes, "Registration decision refers to an unselected dataset")
            if "parent-review" in name:
                require(value["kind"] == "agent_review" and value["human_reviewed"] is False
                        and isinstance(value["records"], list) and value["records"], "Not a structured agent admission decision")
                for decision in value["records"]:
                    require(decision["decision"] in {"admit", "exclude"}
                            and re.fullmatch(r"[a-f0-9]{64}", decision["record_sha256"]), "Unbound admission decision")
            add("registration", ROOT, name)
    for index, folder in enumerate(executions):
        folder = Path(folder).resolve(strict=True)
        execution = runner.verify_execution(folder)
        require(execution["data_sha256"] == provenance["pack_manifest_sha256"], "Execution uses a different prepared pack")
        allowed = {"execution.json"} | {"inputs/" + n for n in manifest["artifacts"]} | {"inputs/manifest.json", "inputs/PREPARED.json"}
        for group, names in (("r3", execution["code_hashes"]), ("reused", execution["reused_code_hashes"]), ("frozen", runner.FROZEN)):
            allowed.update(group + "/" + name for name in names)
        for name in sorted(allowed):
            add("executions/" + str(index), folder, name)
        provenance["executions"].append({"mode": execution["mode"], "execution_sha256": sha(folder / "execution.json")})
    for index, folder in enumerate(runs):
        folder = Path(folder).resolve(strict=True)
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            name = path.relative_to(folder).as_posix()
            try:
                allowed = receipt_allowed(name)
            except ValueError:
                allowed = False
            if allowed:
                value = read(safe_file(folder, name))
                require(isinstance(value, dict) or path.name in {"stages.json", "adapters.json"}, "Invalid training receipt")
                if isinstance(value, dict) and "data_sha256" in value:
                    require(value["data_sha256"] == provenance["pack_manifest_sha256"], "Receipt belongs to another pack")
                add("receipts/" + str(index), folder, name)
                if isinstance(value, dict) and value.get("mode") in {"smoke", "train"}:
                    provenance["run_receipts"].append({"member": "receipts/" + str(index) + "/" + name,
                        **{k: value[k] for k in ("mode", "state", "run_id", "call_id", "remote_directory", "training_completed") if k in value}})
    require(phase != "postrun" or any(r["mode"] == "train" for r in provenance["run_receipts"]), "Postrun snapshot requires an explicit training receipt")
    payload = {}
    for name, path in sorted(paths.items()):
        raw = path.read_bytes()
        screen(name, raw)
        payload[name] = raw
    value = {"schema": SCHEMA, "version": version, "phase": phase, "private": True, "gpu_launched_by_checkpoint_tool": False,
        "phase_meaning": "Before final main-training results; launches and smoke results may exist." if phase == "pre_results"
                         else "Training receipt present; completion is determined only by recorded evidence.",
        "created_at": datetime.now(timezone.utc).isoformat(), "provenance": provenance, "omitted": omitted,
        "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in payload.items()},
        "scope": "Allowlisted research code, contracts, registered synthetic exports, prepared pack and training receipts only.",
        "model_state": "Full model/optimizer checkpoints remain in the private training volume; not duplicated in this metadata archive.",
        "secret_screen_limit": "Allowlist plus byte/JSON credential screening, not a comprehensive secret or personal-data detector.",
        "ledger_copied": False, "user_stores_copied": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r3-freeze-") as temporary:
        staged = Path(temporary)
        save(staged / "manifest.json", value, exclusive=True)
        payload["manifest.json"] = (staged / "manifest.json").read_bytes()
        with (staged / "checkpoint.tar.gz").open("xb") as target, gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name, raw in payload.items():
                    member = tarfile.TarInfo(name)
                    member.size, member.mode, member.mtime = len(raw), 0o600, 0
                    archive.addfile(member, io.BytesIO(raw))
        for name, path in paths.items():
            require(sha(path) == value["files"][name]["sha256"], "Selected source changed during freeze")
        verify(staged)
        os.chmod(staged, 0o700)
        for path in staged.iterdir():
            os.chmod(path, 0o600)
        staged.rename(out)
    return value


def verify(out):
    out = Path(out).resolve(strict=True)
    value = read(safe_file(out, "manifest.json"))
    require(value["schema"] == SCHEMA and value["private"] is True
            and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value["version"]), "Invalid durability manifest")
    expected = set(value["files"]) | {"manifest.json"}
    require({"pack/manifest.json", "pack/PREPARED.json", "pack/rows.jsonl", "pack/source-manifests.json"}
            <= expected, "Missing prepared-pack evidence")
    seen = set()
    with tarfile.open(safe_file(out, "checkpoint.tar.gz"), "r:gz") as archive:
        for member in archive:
            name = safe_name(member.name)
            require(member.isfile() and member_allowed(name) and name not in seen and name in expected, "Unexpected/duplicate archive member")
            seen.add(name)
            raw = archive.extractfile(member).read()
            screen(name, raw)
            if name == "manifest.json":
                require(raw == (out / name).read_bytes(), "Embedded manifest differs")
            else:
                proof = value["files"][name]
                require(len(raw) == proof["bytes"] and hashlib.sha256(raw).hexdigest() == proof["sha256"], "Archive checksum differs: " + name)
                if name == "pack/manifest.json":
                    require(proof["sha256"] == value["provenance"]["pack_manifest_sha256"], "Snapshot pack identity differs")
                    for relative, checksum in json.loads(raw)["artifacts"].items():
                        require(value["files"].get("pack/" + relative, {}).get("sha256") == checksum, "Snapshot pack inventory differs")
    require(seen == expected, "Incomplete archive")
    return value


def remote(out, profile, *, authorize=False, upload=False):
    """Read every remote byte; never overwrite a different version or launch compute."""
    require(not upload or authorize, "Explicit --authorize-upload required")
    out = Path(out).resolve(strict=True)
    manifest = verify(out)
    prefix = PREFIX + "/" + manifest["version"]
    checks = {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
              for name in ("manifest.json", "checkpoint.tar.gz")}
    if (out / "storage.json").exists():
        prior = read(safe_file(out, "storage.json"))
        require(prior["profile"] == profile and prior["volume"] == VOLUME and prior["prefix"] == prefix
                and prior["files"] == checks, "Existing storage receipt differs; do not change a frozen snapshot/profile")
    os.environ["MODAL_PROFILE"] = profile
    import modal
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    missing = []
    def check(name):
        digest, size = hashlib.sha256(), 0
        for chunk in volume.read_file(prefix + "/" + name):
            digest.update(chunk)
            size += len(chunk)
        return {"sha256": digest.hexdigest(), "bytes": size}
    # Check the whole remote version before adding any missing file.
    for name, expected in checks.items():
        try:
            found = check(name)
        except (FileNotFoundError, modal.exception.NotFoundError):
            require(upload, "Remote snapshot incomplete or missing")
            missing.append(name)
        else:
            require(found == expected, "Immutable remote version differs; choose a new version")
    for name in missing:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(out / name, prefix + "/" + name)
    for name, expected in checks.items():
        require(check(name) == expected, "Complete remote checksum readback failed")
    result = {"provider": "modal", "profile": profile, "volume": VOLUME, "prefix": prefix,
        "private": True, "version": manifest["version"], "files": checks,
        "full_remote_readback_verified": True, "gpu_launched_by_checkpoint_tool": False,
        "verified_at": datetime.now(timezone.utc).isoformat()}
    if upload and not (out / "storage.json").exists():
        save(out / "storage.json", result, exclusive=True)
        os.chmod(out / "storage.json", 0o600)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "upload", "readback"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--phase", choices=("pre_results", "prelaunch", "postrun"), default="pre_results",
                        help="pre_results permits already submitted runs; prelaunch is a compatibility alias")
    for name in ("registered-dataset", "execution", "run-root"):
        parser.add_argument("--" + name, action="append", type=Path, default=[])
    parser.add_argument("--profile", default="createdliving1000")
    parser.add_argument("--authorize-upload", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        require(args.pack and args.source and args.version, "Freeze requires --pack, --source, --version")
        value = freeze(args.pack, args.out, args.version, args.source, registered=args.registered_dataset,
                       executions=args.execution, runs=args.run_root, phase=args.phase)
        result = {k: value[k] for k in ("schema", "version", "phase", "provenance", "omitted")}
        result.update(files=len(value["files"]), archive_sha256=sha(args.out / "checkpoint.tar.gz"))
    elif args.action == "verify":
        value = verify(args.out)
        result = {"version": value["version"], "files": len(value["files"]), "archive_sha256": sha(args.out / "checkpoint.tar.gz"), "verified": True}
    else:
        result = remote(args.out, args.profile, authorize=args.authorize_upload, upload=args.action == "upload")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
