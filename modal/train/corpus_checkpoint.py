"""Integrity-checked ephemeral checkpoint staging before vLLM initialization."""
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def stage_checkpoint(source: Path, destination: Path, revision: str):
    if destination.exists(): raise ValueError("Never reuse an unverified partial staging directory")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names or any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
        raise ValueError("Invalid checkpoint shard paths")
    required = sum((source / name).stat().st_size for name in names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required * 1.1:
        raise ValueError("Insufficient ephemeral checkpoint space")
    destination.mkdir()
    started = time.monotonic()

    def copy(name):
        path = source / name
        digest = hashlib.sha256()
        with path.open("rb") as incoming, (destination / name).open("xb") as outgoing:
            while chunk := incoming.read(16 * 1024 * 1024):
                if time.monotonic() - started > 300: raise TimeoutError("Checkpoint staging exceeded five minutes")
                digest.update(chunk); outgoing.write(chunk)
        expected = path.resolve().name
        if len(expected) != 64 or digest.hexdigest() != expected:
            raise ValueError("Public checkpoint content hash differs")
        return {"name": name, "bytes": path.stat().st_size, "sha256": expected}

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            copied = list(pool.map(copy, names))
        metadata = {}
        for path in source.iterdir():
            if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model"):
                target = destination / path.name
                shutil.copyfile(path, target)
                with target.open("rb") as handle: metadata[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        report = {"revision": revision, "bytes": required, "seconds": time.monotonic() - started,
                  "shards": copied, "metadata_sha256": metadata, "persistent_copy": False}
        (destination / "staged.json").write_text(json.dumps(report))
        return report
    except BaseException:
        shutil.rmtree(destination)
        raise
