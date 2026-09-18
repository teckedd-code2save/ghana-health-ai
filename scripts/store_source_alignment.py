"""Checksum-verified private storage for the source-backed alignment handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import modal
from corpus_release import save, sha
from prepare_source_alignment import verify


def store(folder):
    verify(folder)
    seal = json.loads((folder / "READY.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if seal["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("Handoff seal changed")
    files = [*manifest["artifacts"], "manifest.json", "READY.json"]
    if sum((folder / name).stat().st_size for name in files) > 256 * 1024**2:
        raise ValueError("Inspect larger handoffs before uploading")
    volume_name = "ghana-health-understand-train"
    volume = modal.Volume.from_name(volume_name, create_if_missing=False)
    prefix = f"/corpus-training-releases/{folder.name}/{seal['manifest_sha256'][:12]}"
    pending, hashes = [], {name: sha(folder / name) for name in files}
    for name in files:
        try:
            remote = hashlib.sha256()
            for chunk in volume.read_file(prefix + "/" + name): remote.update(chunk)
        except (FileNotFoundError, modal.exception.NotFoundError):
            pending.append(name)
        else:
            if remote.hexdigest() != hashes[name]:
                raise ValueError("Remote artifact differs; never overwrite")
    if pending:
        with volume.batch_upload(force=False) as upload:
            for name in pending: upload.put_file(folder / name, prefix + "/" + name)
    for name in files:
        remote = hashlib.sha256()
        for chunk in volume.read_file(prefix + "/" + name): remote.update(chunk)
        if remote.hexdigest() != hashes[name]:
            raise ValueError("Remote checksum failed")
    save(folder.with_suffix(".storage.json"), {"provider": "modal", "volume": volume_name,
        "path": prefix, "public": False, "training_started": False, "files_sha256": hashes,
        "includes": "Source-backed alignment, separate English retention, manifests and saved review snapshot; no audio or application secrets"})
    print(json.dumps({"verified_files": len(files), "private_volume": volume_name, "path": prefix}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    store(parser.parse_args().release.resolve())
