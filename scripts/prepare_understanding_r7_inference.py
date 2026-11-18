#!/usr/bin/env python3
"""Read back exact Modal artifacts, then seal a private R7/R6/base inference bundle.

collect reads metadata through the Volume API and hashes adapters in a short
CPU-only, read-only Modal container. It never requests a GPU.
prepare is offline; existing bundles and conversations are never modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import r6_inference as r6
import r7_inference as r7

base = r7.base
ROOT = Path(__file__).resolve().parents[1]


def collect(receipt, out, *, receipt_sha, resume=False):
    base.require(r7.is_sha(receipt_sha) and base.sha(receipt) == receipt_sha, "Expected training receipt hash differs")
    remote_root = r7.receipt_root(base.read(receipt))
    import modal
    out = Path(out)
    base.require(resume or not out.exists(), "Use a new readback directory or explicit --resume")
    out.mkdir(parents=True, mode=0o700, exist_ok=resume)
    volume = modal.Volume.from_name(r7.VOLUME, create_if_missing=False)
    hashes = {}

    def download(remote):
        base.require(remote.startswith("/artifacts/"), "Invalid readback root")
        relative = remote.removeprefix("/artifacts/")
        path = base.safe(out, relative)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                for chunk in volume.read_file("/" + relative):
                    stream.write(chunk)
        hashes[remote] = base.sha(path)
        return path

    def fetch(remote):
        path = download(remote)
        return base.read(path), hashes[remote]

    terminal, _ = fetch(remote_root + "/status.json")
    proof = r7.training_proof(base.read(receipt), terminal, fetch)
    adapters = {"r7_response": proof["variant"], "r6_response": {
        "directory": r7.R6_DIRECTORY, "files": {
            "adapter_config.json": r7.R6_CONFIG, "adapter_model.safetensors": r7.R6_WEIGHTS}}}
    expected = dict(hashes)
    for adapter in adapters.values():
        expected.update({adapter["directory"] + "/" + name: checksum for name, checksum in adapter["files"].items()})
    app = modal.App.lookup("ghana-understanding-r7-readback-private", create_if_missing=True)
    sandbox = modal.Sandbox.create(app=app, image=modal.Image.debian_slim(python_version="3.11"),
        cpu=1, memory=512, timeout=180, block_network=True, volumes={"/artifacts": volume.read_only()})
    try:
        process = sandbox.exec("python", "-c", """
import hashlib, json, os, sys
result = {}
for path in json.loads(sys.argv[1]):
    with open(path, "rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    result[path] = {"sha256": checksum, "bytes_read": os.path.getsize(path)}
print(json.dumps(result))
""", json.dumps(sorted(expected)), timeout=150)
        output, error = process.stdout.read(), process.stderr.read()
        base.require(process.wait() == 0, "Modal readback failed: " + error[:1000])
        actual = json.loads(output)
        base.require(set(actual) == set(expected) and all(actual[p]["sha256"] == checksum
            and actual[p]["bytes_read"] > 0 for p, checksum in expected.items()), "Modal readback hash differs")
    finally:
        sandbox.terminate()
    readbacks = {variant: {**adapter, "bytes_read": {
        name: actual[adapter["directory"] + "/" + name]["bytes_read"] for name in adapter["files"]},
        "verified": True, "method": "SHA-256 of full mounted Modal bytes; CPU-only read-only verification"}
        for variant, adapter in adapters.items()}
    base.save(out / "adapter-readback.json", readbacks)
    base.save(out / "modal-readback.json", {"volume": r7.VOLUME, "files": actual,
        "collected_at": datetime.now(timezone.utc).isoformat(), "gpu_called": False})
    return {"readback_root": str(out.resolve()), "files_verified": len(actual), "gpu_called": False}


def prepare(parent_bundle, receipt, readback_root, out, *, receipt_sha, app_name=r7.APP_NAME):
    parent_bundle, readback_root, out = map(Path, (parent_bundle, readback_root, out))
    base.require(not out.exists(), "Use a new immutable R7 bundle directory")
    base.require(r7.is_sha(receipt_sha) and base.sha(receipt) == receipt_sha, "Expected training receipt hash differs")
    base.require(app_name == r7.APP_NAME, "Use the separate private R7 actor")
    remote_root = r7.receipt_root(base.read(receipt))
    parent, _ = r6.bound_bundle(parent_bundle, r7.R6_BUNDLE_SHA)
    r7.check_parent(parent, r7.R6_BUNDLE_SHA)
    local = {}

    def fetch(remote):
        base.require(remote.startswith("/artifacts/"), "Invalid readback path")
        source = base.safe(readback_root, remote.removeprefix("/artifacts/"))
        local[remote] = source
        return base.read(source), base.sha(source)

    terminal, _ = fetch(remote_root + "/status.json")
    proof = r7.training_proof(base.read(receipt), terminal, fetch)
    variants = {"base": None, "r6_response": parent["variants"]["r6_response"], "r7_response": proof["variant"]}
    readbacks = base.read(readback_root / "adapter-readback.json")
    base.require(set(readbacks) == {"r6_response", "r7_response"}, "Missing adapter readback")
    for variant, adapter in variants.items():
        if adapter is None:
            continue
        result = readbacks[variant]
        base.require(result["directory"] == adapter["directory"] and result["files"] == adapter["files"]
            and result["verified"] is True and all(type(result["bytes_read"][name]) is int
            and result["bytes_read"][name] > 0 for name in adapter["files"]), "Actual adapter readback differs")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r7-inference-") as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in parent["files"]:
            if name.startswith(("tokenizer/", "code/")):
                destination = base.safe(staged, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base.safe(parent_bundle, name), destination)
        for source in ("scripts/r7_inference.py", "modal/research_understanding_r7_demo.py"):
            shutil.copyfile(ROOT / source, staged / "code" / Path(source).name)
        shutil.copyfile(parent_bundle / "manifest.json", staged / "r6-manifest.json")
        for remote, source in local.items():
            destination = base.safe(staged, "evidence/readback/" + remote.removeprefix("/artifacts/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(receipt, staged / "evidence/receipt.json")
        base.save(staged / "evidence/terminal.json", terminal)
        base.save(staged / "evidence/adapter-readback.json", readbacks)
        manifest = {"version": r7.VERSION, "model": base.MODEL, "revision": base.REVISION,
            "base_sha256": base.BASE_SHA, "runtime": base.RUNTIME, "decoding": base.DECODING,
            "private": True, "training": False, "default_variant": "r7_response",
            "r6_bundle_sha256": r7.R6_BUNDLE_SHA, "r7_receipt_sha256": receipt_sha,
            "r7_training": proof, "variants": variants,
            "actor": {"app_name": app_name, "class_name": r7.CLASS_NAME},
            "remote_files": {path: base.sha(source) for path, source in local.items()},
            "files": {p.relative_to(staged).as_posix(): base.sha(p) for p in sorted(staged.rglob("*")) if p.is_file()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claims": "Identity and completion only; no semantic, native-language or clinical certification"}
        base.save(staged / "manifest.json", manifest)
        checksum = base.sha(staged / "manifest.json")
        r7.load_bundle(staged, checksum)
        base.save(staged / "SEALED.json", {"manifest_sha256": checksum})
        staged.rename(out)
    return {"bundle": str(out.resolve()), "manifest_sha256": checksum,
            "variants": list(variants), "remote_called": False, "deployed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reader = commands.add_parser("collect")
    reader.add_argument("--receipt", type=Path, required=True)
    reader.add_argument("--receipt-sha", required=True)
    reader.add_argument("--out", type=Path, required=True)
    reader.add_argument("--resume", action="store_true", help="Recheck an interrupted readback; no GPU")
    builder = commands.add_parser("prepare")
    builder.add_argument("--r6-bundle", type=Path, required=True)
    builder.add_argument("--receipt", type=Path, required=True)
    builder.add_argument("--receipt-sha", required=True)
    builder.add_argument("--readback-root", type=Path, required=True)
    builder.add_argument("--out", type=Path, required=True)
    builder.add_argument("--app-name", default=r7.APP_NAME)
    args = parser.parse_args()
    result = (collect(args.receipt, args.out, receipt_sha=args.receipt_sha, resume=args.resume) if args.command == "collect" else
              prepare(args.r6_bundle, args.receipt, args.readback_root, args.out, receipt_sha=args.receipt_sha, app_name=args.app_name))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
