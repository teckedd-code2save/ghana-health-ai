#!/usr/bin/env python3
"""Read only one approved completed R4 run; preserve full bytes and original proofs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import r4_inference as r4


def canonical(path, identifier, physical):
    """Resolve only the declared run, never arbitrary sibling volume directories."""
    root = PurePosixPath("/artifacts/understanding-adaptation-r4") / identifier
    physical = PurePosixPath(physical)
    value = PurePosixPath(path)
    r4.require(value.is_absolute() and ".." not in value.parts and ".." not in physical.parts,
               "Unsafe remote evidence path")
    r4.require(physical.is_absolute() and physical.parts[-2:] == ("understanding-adaptation-r4", identifier),
               "Physical root is not the approved run")
    if value.is_relative_to(physical):
        value = root / value.relative_to(physical)
    r4.require(value.is_relative_to(root) and value != root, "Read outside the approved R4 run")
    return value.as_posix()


def collect(volume, receipt_path, terminal_path, out, identifier=None, expected_weights=None, *, include_pass1=False):
    """The supplied volume needs only read_file; no calls, writes or deployments."""
    out = Path(out)
    r4.require(not out.exists(), "Use a fresh immutable readback directory")
    receipt, terminal = r4.read(receipt_path), r4.read(terminal_path)
    identifier = receipt["run_id"] if identifier is None else identifier
    r4.require(receipt["run_id"] == terminal["run_id"] == identifier
               and receipt["volume"] == r4.VOLUME and receipt["state"] == "submitted"
               and receipt["mode"] == terminal["mode"] == "train"
               and terminal["state"] == "completed" and terminal["training_completed"] is True,
               "Expected the approved completed training run")
    r4.require(expected_weights is None or r4.is_sha(expected_weights), "Invalid optional approved adapter SHA")
    root = "/artifacts/understanding-adaptation-r4/" + identifier
    r4.require(receipt["remote_directory"] == root.removeprefix("/artifacts")
               and terminal["remote_directory"] == root, "Receipt root differs")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r4-readback-") as temporary:
        staged = (Path(temporary) / "snapshot").resolve()
        staged.mkdir()
        downloaded = {}

        def download(path, expected=None):
            remote = canonical(path, identifier, terminal["physical_directory"])
            local = r4.safe(staged, remote.removeprefix("/artifacts/"))
            if remote not in downloaded:
                local.parent.mkdir(parents=True, exist_ok=True)
                checksum, size = hashlib.sha256(), 0
                with local.open("xb") as stream:
                    for chunk in volume.read_file(remote.removeprefix("/artifacts")):
                        size += len(chunk)
                        limit = 512_000_000 if remote.endswith("/adapter_model.safetensors") else 10_000_000
                        r4.require(size <= limit, "Readback exceeds file size bound")
                        stream.write(chunk)
                        checksum.update(chunk)
                r4.require(size > 0 and r4.sha(local) == checksum.hexdigest(), "Empty or incomplete local readback")
                downloaded[remote] = {"sha256": checksum.hexdigest(), "bytes": size,
                                      "path": local.relative_to(staged).as_posix()}
                print(json.dumps({"read": remote, "bytes": size, "sha256": checksum.hexdigest()}), flush=True)
            if expected is not None:
                r4.require(downloaded[remote]["sha256"] == expected, "Readback SHA differs: " + remote)
            return local

        def fetch(path):
            local = download(path)
            return r4.read(local), r4.sha(local)

        proof = r4.training_proof(receipt, terminal, fetch, include_pass1)
        variant = proof["variant"]
        r4.require(expected_weights is None or variant["files"]["adapter_model.safetensors"] == expected_weights,
                   "Final adapter does not match the approved SHA")
        expected_weights = variant["files"]["adapter_model.safetensors"]
        selected = {"responses": variant}
        if include_pass1:
            selected["responses-pass1"] = proof["pass1"]["variant"]
        for stage, spec in selected.items():
            export = next(e for e in terminal["adapters"] if e["stage"] == stage)
            for name, checksum in export["files"].items():
                download(spec["directory"] + "/" + name, checksum)

        consumption = r4.read(download(root + "/consumption.json", terminal["consumption_sha256"]))
        r4.require(consumption["run_id"] == identifier and consumption["mode"] == "train"
                   and consumption["training_complete"] is True and consumption["schedule_complete"] is True
                   and consumption["execution_sha256"] == proof["execution_sha256"]
                   and consumption["data_sha256"] == proof["data_sha256"]
                   and consumption["checkpoint"] == proof["checkpoint"]
                   and consumption["completed_updates"] == proof["completed_updates"], "Consumption proof differs")
        data = r4.read(download(root + "/inputs/manifest.json", proof["data_sha256"]))
        # Metadata only: no training conversations, holdout text or review stores.
        names = ["evaluation-manifest.json", "source-manifests.json", "stages.json"]
        names.extend(ref["manifest_path"] for ref in data["dataset_snapshots"])
        for name in names:
            r4.require(name in data["artifacts"], "Unbound training metadata")
            download(root + "/inputs/" + name, data["artifacts"][name])

        # Re-read the terminal status to detect a concurrently changed remote run.
        current = bytearray()
        for chunk in volume.read_file(receipt["remote_directory"] + "/status.json"):
            current.extend(chunk)
            r4.require(len(current) <= 10_000_000, "Terminal status exceeds size bound")
        saved_status = r4.safe(staged, root.removeprefix("/artifacts/") + "/status.json")
        r4.require(current == saved_status.read_bytes(), "Remote status changed during readback")
        shutil.copyfile(receipt_path, staged / "receipt.json")
        shutil.copyfile(terminal_path, staged / "terminal.json")
        report = {"version": "r4-inference-readback-v1", "run_id": identifier,
                  "volume": r4.VOLUME, "profile": receipt.get("profile"), "remote_directory": root,
                  "receipt_sha256": r4.sha(receipt_path), "terminal_sha256": r4.sha(terminal_path),
                  "adapter_weights_sha256": expected_weights, "completed_updates": proof["completed_updates"],
                  "training_proof": proof, "files": downloaded, "network_operations": ["volume.read_file"],
                  "gpu_called": False, "remote_mutated": False, "proofs_rewritten": False,
                  "checkpoint_state_binary_collected": False,
                  "created_at": datetime.now(timezone.utc).isoformat()}
        r4.base.save(staged / "READBACK.json", report)
        staged.rename(out)
    return {"readback": str(out.resolve()), "run_id": identifier, "files": len(downloaded),
            "bytes": sum(v["bytes"] for v in downloaded.values()),
            "adapter_weights_sha256": expected_weights, "readback_manifest_sha256": r4.sha(out / "READBACK.json"),
            "gpu_called": False, "remote_mutated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "terminal", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", help="Optional additional run identity assertion; defaults to the receipt")
    parser.add_argument("--weights-sha", help="Optional additional final adapter assertion; otherwise verified from completed evidence")
    parser.add_argument("--include-pass1", action="store_true")
    args = parser.parse_args()
    receipt = r4.read(args.receipt)
    r4.require(receipt["volume"] == r4.VOLUME and (args.run_id is None or receipt["run_id"] == args.run_id), "Unapproved volume/run")
    os.environ["MODAL_PROFILE"] = receipt["profile"]
    import modal
    volume = modal.Volume.from_name(r4.VOLUME, create_if_missing=False)
    print(json.dumps(collect(volume, args.receipt, args.terminal, args.out, args.run_id, args.weights_sha,
                             include_pass1=args.include_pass1), indent=2))


if __name__ == "__main__":
    main()
