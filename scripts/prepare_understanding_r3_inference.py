#!/usr/bin/env python3
"""Seal a separate R3 inference bundle from completed, locally read-back evidence.

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

ROOT = Path(__file__).resolve().parents[1]


def prepare(r2_bundle, r2_sha, receipt, terminal, readback_root, out, app_name=r3.APP_NAME):
    r2_bundle, readback_root, out = Path(r2_bundle), Path(readback_root), Path(out)
    r2.require(not out.exists(), "Use a new immutable R3 bundle directory")
    parent = r2.load_bundle(r2_bundle, r2_sha)
    r3.check_parent(parent, r2_sha)
    local = {}

    def fetch(path):
        r2.require(path.startswith("/artifacts/"), "Invalid readback path")
        source = r2.safe(readback_root, path.removeprefix("/artifacts/"))
        local[path] = source
        return r2.read(source), r2.sha(source)

    proof = r3.training_proof(r2.read(receipt), r2.read(terminal), fetch)
    adapter = proof["variant"]
    sizes = {}
    for name, checksum in adapter["files"].items():
        source = r2.safe(readback_root, adapter["directory"].removeprefix("/artifacts/") + "/" + name)
        r2.require(r2.sha(source) == checksum and source.stat().st_size > 0, "Actual R3 adapter readback differs: " + name)
        sizes[name] = source.stat().st_size
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r3-inference-") as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in parent["files"]:
            if name.startswith("tokenizer/") or (name.startswith("code/") and name.endswith(".py")):
                destination = r2.safe(staged, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(r2.safe(r2_bundle, name), destination)
        for name in ("r2_inference.py", "r3_inference.py"):
            shutil.copyfile(ROOT / "scripts" / name, staged / "code" / name)
        shutil.copyfile(r2_bundle / "manifest.json", staged / "r2-manifest.json")
        for remote, source in local.items():
            destination = r2.safe(staged, "evidence/readback/" + remote.removeprefix("/artifacts/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(receipt, staged / "evidence/receipt.json")
        shutil.copyfile(terminal, staged / "evidence/terminal.json")
        r2.save(staged / "evidence/adapter-readback.json", {"directory": adapter["directory"],
            "files": adapter["files"], "bytes_read": sizes, "verified": True,
            "method": "SHA-256 of full local readback bytes; runtime rechecks mounted adapter bytes"})
        manifest = {"version": r3.VERSION, "model": r2.MODEL, "revision": r2.REVISION,
            "base_sha256": r2.BASE_SHA, "runtime": r2.RUNTIME, "decoding": r2.DECODING,
            "private": True, "training": False, "default_variant": "r3_response",
            "r2_bundle_sha256": r2_sha, "r3_training": proof,
            "variants": {"base": None, "r2_meaning": parent["variants"]["r2_meaning"], "r3_response": adapter},
            "actor": {"app_name": app_name, "class_name": r3.CLASS_NAME},
            "remote_files": {**parent["remote_files"], **{path: r2.sha(source) for path, source in local.items()}},
            "files": {p.relative_to(staged).as_posix(): r2.sha(p) for p in sorted(staged.rglob("*")) if p.is_file()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claims": "Completion and identity only; no semantic, native-language or clinical certification"}
        r2.save(staged / "manifest.json", manifest)
        checksum = r2.sha(staged / "manifest.json")
        r3.load_bundle(staged, checksum)
        r2.save(staged / "SEALED.json", {"manifest_sha256": checksum})
        staged.rename(out)
    return {"bundle": str(out.resolve()), "manifest_sha256": checksum,
            "variant": "r3_response", "adapter_weights_sha256": adapter["files"]["adapter_model.safetensors"],
            "remote_called": False, "deployed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-bundle", type=Path, required=True)
    parser.add_argument("--r2-sha", default=r3.R2_BUNDLE_SHA)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--readback-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--app-name", default=r3.APP_NAME)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.r2_bundle, args.r2_sha, args.receipt, args.terminal,
                             args.readback_root, args.out, args.app_name), indent=2))


if __name__ == "__main__":
    main()
