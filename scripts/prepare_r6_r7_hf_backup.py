"""Prepare and verify private checkpoint backups. No cloud compute or corpus upload."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_r7_local_mlx import BASE_SHA, MODEL, REVISION, VARIANTS, require, sha

R7_RUN = "understanding-adaptation-r7/understanding_20261118T010701Z"
RECOVERY_SHA = "2fb8e9310d2c889316ae6a066efb9d18b863edd305d13a1115db1cfbb0221a23"
R7_DATA_SHA = "999d53cdbc79f4d0268edf4e2f9ec270756155f3341c81069deec82ad8243d53"
R7_RUN_SHA = "2e61220235dc18f1ed98936c687509146670b1d6842fd32c697add84a6197c07"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(export, bundle, output, r7_inference_only=False):
    from check_afrique_training_contract import ASSET_HASHES
    require(not output.exists(), "Use a new immutable output directory")
    output.mkdir(parents=True, mode=0o700)
    for version in (("r7",) if r7_inference_only else ("r6", "r7")):
        relative, weights_sha, config_sha = VARIANTS[version]
        source = export / relative
        require(sha(source / "adapter_model.safetensors") == weights_sha, "Wrong adapter")
        require(sha(source / "adapter_config.json") == config_sha, "Wrong configuration")
        destination = output / version
        destination.mkdir()
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copyfile(source / name, destination / name)
        (destination / "tokenizer").mkdir()
        for name, checksum in ASSET_HASHES.items():
            require(sha(bundle / "tokenizer" / name) == checksum, "Changed frozen tokenizer")
            shutil.copyfile(bundle / "tokenizer" / name, destination / "tokenizer" / name)
        recovery = {"included": False, "reason": "R6 optimizer/RNG state is retained on Modal, not in this local export"}
        if version == "r7" and r7_inference_only:
            recovery = {"included": False, "state_sha256": RECOVERY_SHA,
                        "reason": "Full recovery state remains in local and Modal archives; HF recovery upload pending"}
        elif version == "r7":
            state = export / R7_RUN / "responses/attempt-226621fo/update-00002413"
            require(sha(state / "state.pt") == RECOVERY_SHA, "Wrong R7 recovery state")
            manifest = json.loads((state / "manifest.json").read_text())
            committed = json.loads((state / "COMMITTED.json").read_text())
            require(sha(state / "manifest.json") == committed["manifest_sha256"] and
                    manifest["state_sha256"] == RECOVERY_SHA, "Uncommitted recovery state")
            (destination / "recovery/update-00002413").mkdir(parents=True)
            for name in ("state.pt", "manifest.json", "COMMITTED.json"):
                shutil.copyfile(state / name, destination / "recovery/update-00002413" / name)
            recovery = {"included": True, "update": 2413, "state_sha256": RECOVERY_SHA,
                        "requires_original_training_code_and_sealed_data": True,
                        "resume_test_from_hf": False}
        write(destination / "SOURCE_IDENTITY.json", {
            "variant": version, "base_model": MODEL, "base_revision": REVISION,
            "base_weights_sha256": BASE_SHA, "adapter_sha256": weights_sha,
            "adapter_config_sha256": config_sha, "recovery": recovery,
            "standalone_full_model": False, "base_weights_included": False,
            "corpus_included": False, "paid_compute_launched": False,
            "backup_scope": "inference_only" if r7_inference_only else "original_export",
            "restriction": "Private research backup; not a clinical or commercial certification"})
        data_note = ("R6 continued R5. The registered pack contains 6,290 records and 8,373 views; "
                     "7,902 distinct views were consumed across 9,940 presentations and 622 updates."
                     if version == "r6" else
                     "R7 continued R6. All 38,607 scheduled views were consumed across 2,413 updates, "
                     "with 1,849,881 effective target tokens. Views are not independent conversations.")
        evaluation = ("""Corrected matched results, correct answers / questions:

| Task | Base | R5 | R6 |
| --- | ---: | ---: | ---: |
| AfriMMLU Twi, three subjects | 79/300 | 112/300 | 120/300 |
| SIB-200 Twi | 154/204 | 160/204 | 166/204 |
| MMLU English, three subjects | 164/300 | 201/300 | 197/300 |
| HellaSwag English | 179/300 | 200/300 | 199/300 |
| ARC-Challenge English | 265/300 | 280/300 | 279/300 |
| MedQA English | 118/300 | 198/300 | 193/300 |
| PubMedQA English | 187/300 | 175/300 | 174/300 |

These are custom zero-shot source-order subsets, not leaderboard scores.
MCQs use conditional log-probability of the presented choice letters; SIB uses
greedy generation with strict single-label parsing. Thinking is disabled for all.
HellaSwag uses a nonstandard choice-letter protocol. PubMedQA uses its labeled
training partition as an evaluation source; overlap is not comprehensively
certified. R6 improves these two Twi aggregates over R5 but all five English
aggregates fall slightly. The Twi paired tests do not reach p < 0.05. These
results do not establish conversational usefulness or medical safety.

Full protocol, corrections, limitations and preserved historical results:
[R6-STATUS.md](https://github.com/teckedd-code2save/ghana-health-ai/blob/a41488f/R6-STATUS.md).
""" if version == "r6" else """Training completed. Matched R7 quality evaluation is pending; no improvement
over R6 is claimed. Training completion and checkpoint verification are not
quality benchmarks.

| Collection | Consumed training views |
| --- | ---: |
| Bilingual alignment and meaning | 31,902 |
| Direct responses | 2,644 |
| English retention | 2,554 |
| Language replay | 1,187 |
| Executed English tools | 320 |
| Total | 38,607 |

There are 17,856 distinct source IDs, not 38,607 independent conversations.
Follow-up views reuse original Twi targets with synthetic English instructions.
Tool histories contain actual sandbox results against a fictional catalogue;
they are not real shopping or evidence of deployed web search. Bulk generated
Twi dialogue and new medical expansion were deferred after teacher failures.
Data manifest SHA256: `""" + R7_DATA_SHA + """`.

Sources, admissions and remaining gaps:
[R7 dataset card](https://github.com/teckedd-code2save/ghana-health-ai/blob/a41488f/docs/research/R7_DATASET_CARD.md),
[execution record](https://github.com/teckedd-code2save/ghana-health-ai/blob/a41488f/R7-STATUS.md).
""")
        recovery_note = (
            "This is an inference-only checkpoint backup. The R7 optimizer, scheduler "
            "and RNG recovery state is NOT uploaded here. It remains in the original "
            "local and Modal archives; its HF upload is pending. Restoring this adapter "
            "does not constitute restoring the full training state."
            if r7_inference_only else
            "For R7, `recovery/update-00002413/` includes the adapter/optimizer/scheduler/RNG "
            "state and committed manifest. Resuming also requires the original frozen code "
            "and sealed dataset, retained in the private research archive. R6 full optimizer "
            "state is not included in this export."
        )
        card = f"""---
language:
- tw
- en
library_name: peft
pipeline_tag: text-generation
base_model: {MODEL}
tags:
- lora
- research
---
# Ghana Health AI {version.upper()}: Private Checkpoint Backup

This repository contains actual LoRA weights, not just a model card. It is a
private research backup. Source licenses and corpus restrictions still apply;
this backup grants no additional redistribution or commercial-use rights.

## Model and Training

{data_note}

The frozen foundation is approximately 9B parameters. The adapter has
43,278,336 parameters, rank 16 and alpha 32. Load the pinned foundation with
exactly this ONE adapter. Do not stack R6 and R7: R7 already continues R6.

Base revision: `{REVISION}`. Adapter SHA256: `{weights_sha}`.
The full base weights are not duplicated here; they remain a pinned dependency.
`tokenizer/` preserves the four frozen tokenizer/template/config assets.

## Evidence and Limitations

{evaluation}

The model can still misinterpret Twi, repeat itself, provide unsupported facts,
fail to follow corrections or produce unsafe advice. It is not clinically
validated and is not a medical service. No stronger model replaces its replies.
Training-data records are not uploaded here. Aggregate counts and source
provenance are documented in the project research dataset cards.

## Restore and Verify

1. Download this repository at the exact commit recorded in the upload receipt.
2. Check every file against `CHECKSUMS.json` before loading weights.
3. Download `{MODEL}` at revision `{REVISION}`. Its weight SHA256 must be
   `{BASE_SHA}`. Preserve this base separately for independence from its publisher.
4. Use the frozen tokenizer and native chat template with thinking disabled.
   Load `adapter_model.safetensors` with the matching PEFT configuration.
5. For an Apple Silicon comparison, follow `docs/research/R7_LOCAL_MAC.md` in
   https://github.com/teckedd-code2save/ghana-health-ai on branch
   `codex/61-r7-data-training`. Quantized scores are separate from BF16 scores.

`SOURCE_IDENTITY.json` distinguishes inference restoration from training resume.
{recovery_note}
This is not a new training authorization.
Never load arbitrary pickle checkpoints; the R7 state is an owner-created,
hash-verified artifact. No HF restore inference or training-resume test is claimed
until separately recorded. Keep the existing local and Modal copies.
"""
        (destination / "README.md").write_text(card)
        files = {p.relative_to(destination).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(destination.rglob("*")) if p.is_file()}
        write(destination / "CHECKSUMS.json", {"version": 1, "variant": version, "files": files,
              "backup_scope": "inference_only" if r7_inference_only else "original_export"})
        print(json.dumps({"prepared": version, "bytes": sum(f["bytes"] for f in files.values()),
                          "manifest_sha256": sha(destination / "CHECKSUMS.json")}), flush=True)


def verify_package(folder):
    from check_afrique_training_contract import ASSET_HASHES
    manifest = json.loads((folder / "CHECKSUMS.json").read_text())
    require(manifest["variant"] in VARIANTS and manifest["version"] == 1, "Unknown package")
    scope = manifest.get("backup_scope", "original_export")
    require(scope in ("original_export", "inference_only"), "Unknown backup scope")
    require(scope != "inference_only" or manifest["variant"] == "r7", "Unexpected inference-only variant")
    allowed = {"adapter_model.safetensors", "adapter_config.json", "README.md", "SOURCE_IDENTITY.json"}
    allowed.update("tokenizer/" + name for name in ASSET_HASHES)
    if manifest["variant"] == "r7" and scope != "inference_only":
        allowed.update("recovery/update-00002413/" + name for name in ("state.pt", "manifest.json", "COMMITTED.json"))
    require(set(manifest["files"]) == allowed, "Non-checkpoint content is not authorized for upload")
    expected = set(manifest["files"]) | {"CHECKSUMS.json"}
    require(expected == {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()},
            "Unexpected or missing package files")
    for name, proof in manifest["files"].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()) and not (folder / name).is_symlink(), "Escaping backup path")
        require(path.stat().st_size == proof["bytes"] and sha(path) == proof["sha256"], "Changed backup file: " + name)
    require(sha(folder / "adapter_model.safetensors") == VARIANTS[manifest["variant"]][1], "Unrecognized adapter")
    require(sha(folder / "adapter_config.json") == VARIANTS[manifest["variant"]][2], "Unrecognized adapter config")
    for name, checksum in ASSET_HASHES.items():
        require(sha(folder / "tokenizer" / name) == checksum, "Changed pinned tokenizer asset: " + name)
    identity = json.loads((folder / "SOURCE_IDENTITY.json").read_text())
    require((identity["variant"], identity["base_model"], identity["base_revision"],
             identity["base_weights_sha256"], identity["adapter_sha256"], identity["adapter_config_sha256"]) ==
            (manifest["variant"], MODEL, REVISION, BASE_SHA, *VARIANTS[manifest["variant"]][1:]),
            "Changed source identity")
    require(identity["corpus_included"] is False and identity["base_weights_included"] is False,
            "Invalid backup scope")
    require(identity.get("backup_scope", "original_export") == scope, "Conflicting backup scope")
    if scope == "inference_only":
        require(identity["recovery"]["included"] is False and
                identity["recovery"]["state_sha256"] == RECOVERY_SHA, "Invalid inference-only recovery identity")
    elif manifest["variant"] == "r7":
        recovery = folder / "recovery/update-00002413"
        state = json.loads((recovery / "manifest.json").read_text())
        committed = json.loads((recovery / "COMMITTED.json").read_text())
        require(sha(recovery / "state.pt") == RECOVERY_SHA and state["state_sha256"] == RECOVERY_SHA,
                "Wrong recovery state")
        require(state["identity"] == {"data_sha256": R7_DATA_SHA, "run_sha256": R7_RUN_SHA} and
                state["state_bytes"] == (recovery / "state.pt").stat().st_size and
                committed["manifest_sha256"] == sha(recovery / "manifest.json"), "Invalid recovery binding")
    return manifest


def upload(folder, receipt):
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import RepositoryNotFoundError

    require(not receipt.exists(), "Use a new receipt path")
    manifest = verify_package(folder)
    require(get_token(), "Sign in locally with a write-capable HF token; do not put tokens in arguments")
    api = HfApi()
    account = api.whoami()
    require(account["name"] == "teckedd", "Unexpected HF account")
    require(account.get("isPro") is False, "Recheck paid-account storage headroom before uploading")
    total = sum(p["bytes"] for p in manifest["files"].values())
    require(total < 1024**3, "Unexpectedly large package; recheck storage budget")
    repo = "teckedd/gha-twi-understanding-" + manifest["variant"]
    try:
        info = api.model_info(repo)
        require(info.private is True, "Refuse to put private backup files into a public repository")
    except RepositoryNotFoundError:
        api.create_repo(repo, repo_type="model", private=True)
    # Free account's quota must accept this upload. Never buy storage or change plans.
    require(api.model_info(repo).private is True, "Repository privacy not verified")
    commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=folder,
                               commit_message="Back up verified final checkpoint and recovery metadata")
    print(json.dumps({"uploaded_repo": repo, "revision": commit.oid, "verification_pending": True}), flush=True)
    verify_remote(folder, receipt, commit.oid, api=api)


def verify_remote(folder, receipt, revision, api=None):
    from huggingface_hub import HfApi, hf_hub_download
    import re

    require(not receipt.exists(), "Use a new receipt path")
    require(re.fullmatch(r"[a-f0-9]{40}", revision), "Verification needs an exact HF commit")
    manifest = verify_package(folder)
    api = api or HfApi()
    require(api.whoami()["name"] == "teckedd", "Unexpected HF account")
    repo = "teckedd/gha-twi-understanding-" + manifest["variant"]
    info = api.model_info(repo, revision=revision, files_metadata=True)
    require(info.private is True, "Unexpected public backup")
    remote = {p.rfilename: p for p in info.siblings}
    expected = {**manifest["files"], "CHECKSUMS.json": {"sha256": sha(folder / "CHECKSUMS.json"),
                                                         "bytes": (folder / "CHECKSUMS.json").stat().st_size}}
    checked = {}
    for name, proof in expected.items():
        require(name in remote and remote[name].size == proof["bytes"], "Missing/incorrect remote size: " + name)
        item = remote[name]
        if item.lfs:
            require(item.lfs.sha256 == proof["sha256"], "Remote LFS SHA mismatch: " + name)
            method = "remote_lfs_sha256"
        else:
            downloaded = hf_hub_download(repo, name, revision=revision, token=True)
            require(sha(downloaded) == proof["sha256"], "Readback SHA mismatch: " + name)
            method = "downloaded_sha256"
        checked[name] = {**proof, "verification": method}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    require(not receipt.exists(), "Use a new receipt path")
    write(receipt, {"repo": repo, "private": True, "revision": revision, "files": checked,
                    "model_weights_present": True, "remote_checksums_verified": True,
                    "full_weight_roundtrip_download": False, "full_restore_inference_test": False,
                    "corpus_uploaded": False, "cloud_compute_launched": False,
                    "paid_plan_or_storage_enabled": False})
    print(json.dumps({"repo": repo, "revision": revision, "verified_files": len(checked)}), flush=True)


def restore_check(folder, output, receipt, revision, resume=False):
    from huggingface_hub import HfApi, snapshot_download
    import re

    require(not receipt.exists(), "Use a new receipt path")
    require(resume or not output.exists(), "Use a new restore path or --resume")
    require(re.fullmatch(r"[a-f0-9]{40}", revision), "Restore needs an exact HF commit")
    manifest = verify_package(folder)
    repo = "teckedd/gha-twi-understanding-" + manifest["variant"]
    binding = {"repo": repo, "revision": revision, "package_sha256": sha(folder / "CHECKSUMS.json")}
    if resume:
        require(json.loads((output / "restore-request.json").read_text()) == binding,
                "Restore request changed; cannot resume")
    info = HfApi().model_info(repo, revision=revision)
    require(info.private is True and info.sha == revision, "Private committed checkpoint not verified")
    expected = {**manifest["files"], "CHECKSUMS.json": {
        "bytes": (folder / "CHECKSUMS.json").stat().st_size, "sha256": sha(folder / "CHECKSUMS.json")}}
    # A fresh local cache verifies restoration through HF rather than reusing the source package.
    output.mkdir(parents=True, mode=0o700, exist_ok=resume)
    if not resume:
        write(output / "restore-request.json", binding)
    snapshot = Path(snapshot_download(repo, revision=revision, cache_dir=output,
                                     allow_patterns=list(expected), token=True, max_workers=1))
    checked = {}
    for name, proof in expected.items():
        path = (snapshot / name).resolve()
        require(path.is_relative_to(output.resolve()), "Escaping restored path")
        require(path.stat().st_size == proof["bytes"] and sha(path) == proof["sha256"],
                "Restored file mismatch: " + name)
        checked[name] = proof
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write(receipt, {"repo": repo, "revision": revision, "private": True,
                    "snapshot": str(snapshot), "files": checked, "resumed": resume,
                    "full_checkpoint_download_and_hash_verified": True,
                    "restore_inference_test": False, "training_resume_test": False,
                    "cloud_compute_launched": False})
    print(json.dumps({"restored": repo, "revision": revision, "verified_files": len(checked)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("--export", type=Path, required=True)
    make.add_argument("--bundle", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--r7-inference-only", action="store_true",
                      help="Prepare only R7 inference files; keep full recovery in its original archives")
    send = commands.add_parser("upload")
    send.add_argument("--folder", type=Path, required=True)
    send.add_argument("--receipt", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--folder", type=Path, required=True)
    check.add_argument("--receipt", type=Path, required=True)
    check.add_argument("--revision", required=True)
    restore = commands.add_parser("restore-check")
    restore.add_argument("--folder", type=Path, required=True)
    restore.add_argument("--out", type=Path, required=True)
    restore.add_argument("--receipt", type=Path, required=True)
    restore.add_argument("--revision", required=True)
    restore.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.export, args.bundle, args.out, args.r7_inference_only)
    elif args.command == "upload":
        upload(args.folder, args.receipt)
    elif args.command == "verify":
        verify_remote(args.folder, args.receipt, args.revision)
    else:
        restore_check(args.folder, args.out, args.receipt, args.revision, args.resume)


if __name__ == "__main__":
    from huggingface_hub.utils import logging as hub_logging
    # Upload exceptions can contain presigned URLs. Never print their messages or chains.
    hub_logging.set_verbosity_error()
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "detail": "No success claimed. Check private repository state before resuming."}), flush=True)
        raise SystemExit(1) from None
