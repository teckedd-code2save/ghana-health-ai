#!/usr/bin/env python3
"""Read back exact Modal artifacts, then seal a private R6/R5/base inference bundle.

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

import r5_inference as r5
import r6_inference as r6

base = r6.base
ROOT = Path(__file__).resolve().parents[1]


def collect(receipt, out, *, resume=False):
    import modal
    out = Path(out)
    base.require(resume or not out.exists(), "Use a new readback directory or explicit --resume")
    out.mkdir(parents=True, mode=0o700, exist_ok=resume)
    volume = modal.Volume.from_name(r6.VOLUME, create_if_missing=False)
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

    terminal, _ = fetch(str(Path(r6.R6_DIRECTORY).parent) + "/status.json")
    proof = r6.training_proof(base.read(receipt), terminal, fetch)
    adapters = {"r6_response": proof["variant"], "r5_response": {
        "directory": r6.R5_DIRECTORY, "files": {
            "adapter_config.json": r6.R5_CONFIG, "adapter_model.safetensors": r6.R5_WEIGHTS}}}
    expected = dict(hashes)
    for adapter in adapters.values():
        expected.update({adapter["directory"] + "/" + name: checksum for name, checksum in adapter["files"].items()})
    app = modal.App.lookup("ghana-understanding-r6-readback-private", create_if_missing=True)
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
    base.save(out / "modal-readback.json", {"volume": r6.VOLUME, "files": actual,
        "collected_at": datetime.now(timezone.utc).isoformat(), "gpu_called": False})
    return {"readback_root": str(out.resolve()), "files_verified": len(actual), "gpu_called": False}


def prepare(parent_bundle, receipt, readback_root, out, *, app_name=r6.APP_NAME):
    parent_bundle, readback_root, out = map(Path, (parent_bundle, readback_root, out))
    base.require(not out.exists(), "Use a new immutable R6 bundle directory")
    parent = r5.load_bundle(parent_bundle, r6.R5_BUNDLE_SHA)
    r6.check_parent(parent, r6.R5_BUNDLE_SHA)
    local = {}

    def fetch(remote):
        base.require(remote.startswith("/artifacts/"), "Invalid readback path")
        source = base.safe(readback_root, remote.removeprefix("/artifacts/"))
        local[remote] = source
        return base.read(source), base.sha(source)

    terminal, _ = fetch(str(Path(r6.R6_DIRECTORY).parent) + "/status.json")
    proof = r6.training_proof(base.read(receipt), terminal, fetch)
    variants = {"base": None, "r5_response": parent["variants"]["r5_response"], "r6_response": proof["variant"]}
    readbacks = base.read(readback_root / "adapter-readback.json")
    base.require(set(readbacks) == {"r5_response", "r6_response"}, "Missing adapter readback")
    for variant, adapter in variants.items():
        if adapter is None:
            continue
        result = readbacks[variant]
        base.require(result["directory"] == adapter["directory"] and result["files"] == adapter["files"]
            and result["verified"] is True and all(type(result["bytes_read"][name]) is int
            and result["bytes_read"][name] > 0 for name in adapter["files"]), "Actual adapter readback differs")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r6-inference-") as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in parent["files"]:
            if name.startswith(("tokenizer/", "code/")):
                destination = base.safe(staged, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base.safe(parent_bundle, name), destination)
        for source in ("scripts/r6_inference.py", "modal/research_understanding_r6_demo.py"):
            shutil.copyfile(ROOT / source, staged / "code" / Path(source).name)
        shutil.copyfile(parent_bundle / "manifest.json", staged / "r5-manifest.json")
        for remote, source in local.items():
            destination = base.safe(staged, "evidence/readback/" + remote.removeprefix("/artifacts/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(receipt, staged / "evidence/receipt.json")
        base.save(staged / "evidence/terminal.json", terminal)
        base.save(staged / "evidence/adapter-readback.json", readbacks)
        manifest = {"version": r6.VERSION, "model": base.MODEL, "revision": base.REVISION,
            "base_sha256": base.BASE_SHA, "runtime": base.RUNTIME, "decoding": base.DECODING,
            "private": True, "training": False, "default_variant": "r6_response",
            "r5_bundle_sha256": r6.R5_BUNDLE_SHA, "r6_training": proof, "variants": variants,
            "actor": {"app_name": app_name, "class_name": r6.CLASS_NAME},
            "remote_files": {path: base.sha(source) for path, source in local.items()},
            "files": {p.relative_to(staged).as_posix(): base.sha(p) for p in sorted(staged.rglob("*")) if p.is_file()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claims": "Identity and completion only; no semantic, native-language or clinical certification"}
        base.save(staged / "manifest.json", manifest)
        checksum = base.sha(staged / "manifest.json")
        r6.load_bundle(staged, checksum)
        base.save(staged / "SEALED.json", {"manifest_sha256": checksum})
        staged.rename(out)
    return {"bundle": str(out.resolve()), "manifest_sha256": checksum,
            "variants": list(variants), "remote_called": False, "deployed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reader = commands.add_parser("collect")
    reader.add_argument("--receipt", type=Path, required=True)
    reader.add_argument("--out", type=Path, required=True)
    reader.add_argument("--resume", action="store_true", help="Recheck an interrupted readback; no GPU")
    builder = commands.add_parser("prepare")
    builder.add_argument("--r5-bundle", type=Path, required=True)
    builder.add_argument("--receipt", type=Path, required=True)
    builder.add_argument("--readback-root", type=Path, required=True)
    builder.add_argument("--out", type=Path, required=True)
    builder.add_argument("--app-name", default=r6.APP_NAME)
    args = parser.parse_args()
    result = (collect(args.receipt, args.out, resume=args.resume) if args.command == "collect" else
              prepare(args.r5_bundle, args.receipt, args.readback_root, args.out, app_name=args.app_name))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
