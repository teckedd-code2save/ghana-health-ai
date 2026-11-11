#!/usr/bin/env python3
"""Seal a separate R4 inference bundle from completed, locally read-back evidence.

Offline only: no Modal import, collection, deployment or inference. Readback root
mirrors /artifacts; the owner supplies bytes collected after actual completion.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

import r2_inference as r2
import r3_inference as r3
import r4_inference as r4

ROOT = Path(__file__).resolve().parents[1]


def prepare(r3_bundle, r3_sha, receipt, terminal, readback_root, out, *, helpers_root,
            app_name=r4.APP_NAME, include_pass1=False):
    r3_bundle, readback_root, out = Path(r3_bundle), Path(readback_root), Path(out)
    r2.require(not out.exists(), "Use a new immutable R4 bundle directory")
    parent = r3.load_bundle(r3_bundle, r3_sha)
    r4.check_parent(parent, r3_sha)
    helpers_root = Path(helpers_root)
    for name, checksum in r4.HELPER_HASHES.items():
        r2.require(r2.sha(r2.safe(helpers_root, "scripts/" + name)) == checksum,
                   "Pinned streaming/guard helper changed: " + name)
    local = {}

    def fetch(path):
        r2.require(path.startswith("/artifacts/"), "Invalid readback path")
        source = r2.safe(readback_root, path.removeprefix("/artifacts/"))
        local[path] = source
        return r2.read(source), r2.sha(source)

    proof = r4.training_proof(r2.read(receipt), r2.read(terminal), fetch, include_pass1)
    adapter = proof["variant"]
    variants = {"base": None, "r3_response": parent["variants"]["r3_response"], "r4_response": adapter}
    if include_pass1:
        variants["r4_pass1"] = proof["pass1"]["variant"]
    readbacks = {}
    for variant, spec in variants.items():
        if not variant.startswith("r4_"):
            continue
        sizes = {}
        for name, checksum in spec["files"].items():
            source = r2.safe(readback_root, spec["directory"].removeprefix("/artifacts/") + "/" + name)
            r2.require(r2.sha(source) == checksum and source.stat().st_size > 0, "Actual R4 adapter readback differs: " + name)
            sizes[name] = source.stat().st_size
        readbacks[variant] = {**spec, "bytes_read": sizes, "verified": True,
            "method": "SHA-256 of full local readback bytes; runtime rechecks mounted adapter bytes"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r4-inference-") as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in parent["files"]:
            if name.startswith("tokenizer/") or (name.startswith("code/") and name.endswith(".py")):
                destination = r2.safe(staged, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(r2.safe(r3_bundle, name), destination)
        shutil.copyfile(ROOT / "scripts/r4_inference.py", staged / "code/r4_inference.py")
        shutil.copyfile(ROOT / "modal/research_understanding_r4_demo.py", staged / "code/research_understanding_r4_demo.py")
        for name in r4.HELPER_HASHES:
            shutil.copyfile(helpers_root / "scripts" / name, staged / "code" / name)
        shutil.copyfile(r3_bundle / "manifest.json", staged / "r3-manifest.json")
        for remote, source in local.items():
            destination = r2.safe(staged, "evidence/readback/" + remote.removeprefix("/artifacts/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(receipt, staged / "evidence/receipt.json")
        shutil.copyfile(terminal, staged / "evidence/terminal.json")
        r2.save(staged / "evidence/adapter-readback.json", readbacks)
        manifest = {"version": r4.VERSION, "model": r2.MODEL, "revision": r2.REVISION,
            "base_sha256": r2.BASE_SHA, "runtime": r2.RUNTIME, "decoding": r2.DECODING,
            "private": True, "training": False, "default_variant": "r4_response",
            "r3_bundle_sha256": r3_sha, "r4_training": proof,
            "variants": variants,
            "actor": {"app_name": app_name, "class_name": r4.CLASS_NAME},
            "remote_files": {**parent["remote_files"], **{path: r2.sha(source) for path, source in local.items()}},
            "files": {p.relative_to(staged).as_posix(): r2.sha(p) for p in sorted(staged.rglob("*")) if p.is_file()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claims": "Completion and identity only; no semantic, native-language or clinical certification"}
        r2.save(staged / "manifest.json", manifest)
        checksum = r2.sha(staged / "manifest.json")
        r4.load_bundle(staged, checksum)
        r2.save(staged / "SEALED.json", {"manifest_sha256": checksum})
        staged.rename(out)
    return {"bundle": str(out.resolve()), "manifest_sha256": checksum,
            "variant": "r4_response", "variants": list(variants),
            "adapter_weights_sha256": adapter["files"]["adapter_model.safetensors"],
            "remote_called": False, "deployed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-bundle", type=Path, required=True)
    parser.add_argument("--r3-sha", default=r4.R3_BUNDLE_SHA)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--readback-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--app-name", default=r4.APP_NAME)
    parser.add_argument("--helpers-root", type=Path, required=True,
                        help="Source tree containing the pinned guarded R3 streaming helpers")
    parser.add_argument("--include-pass1", action="store_true",
                        help="Require the pass-one export and committed checkpoint within this completed run")
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.r3_bundle, args.r3_sha, args.receipt, args.terminal,
                             args.readback_root, args.out, helpers_root=args.helpers_root,
                             app_name=args.app_name, include_pass1=args.include_pass1), indent=2))


if __name__ == "__main__":
    main()
