#!/usr/bin/env python3
"""Content-addressed private research backup, with remote read-back verification."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "/research-handoffs/twi-understanding/2026-11-17-r7"
PATTERNS = ("R7-STATUS.md", "scripts/*r7*.py", "modal/train/*r7*.py", "modal/*r7*.py", "docs/research/R7*.md")
EXCLUDED_RUNTIME = {"baseline-launch-py311", "__pycache__", ".venv"}
SECRET = re.compile(rb"sk-proj-[A-Za-z0-9_-]{16,}|OPENAI_API_KEY\s*=\s*['\"]?[A-Za-z0-9_-]{16,}")


class CheckedReader:
    def __init__(self, stream):
        self.stream, self.tail = stream, b""
        self.digest = hashlib.sha256()

    def read(self, size):
        value = self.stream.read(size)
        if SECRET.search(self.tail + value):
            raise ValueError("Potential credential in backup input")
        self.tail = (self.tail + value)[-4096:]
        self.digest.update(value)
        return value


def build(out):
    root = ROOT.resolve(strict=True)
    paths = set()
    for pattern in PATTERNS:
        paths.update(root.glob(pattern))
    data = root / "tmp/r7-20261117"
    paths.update(p for p in data.rglob("*") if p.suffix in {".json", ".jsonl", ".md"}
                 and "backups" not in p.parts and not EXCLUDED_RUNTIME.intersection(p.relative_to(data).parts)
                 and p.is_file())
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "private": True,
                "status": "work_in_progress_not_training_release",
                "excluded_recreatable_runtime_directories": sorted(EXCLUDED_RUNTIME),
                "files": {}}
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "snapshot.tmp"
    try:
        with temporary.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(paths):
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        raise ValueError("Linked backup input")
                    name = path.relative_to(root).as_posix()
                    entry = tarfile.TarInfo(name)
                    with path.open("rb") as source:
                        import os
                        before = os.fstat(source.fileno())
                        entry.size, entry.mode, entry.mtime = before.st_size, 0o600, 0
                        reader = CheckedReader(source)
                        archive.addfile(entry, reader)
                        after = os.fstat(source.fileno())
                        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                            raise ValueError("Backup input changed while being read: " + name)
                        manifest["files"][name] = reader.digest.hexdigest()
                raw = json.dumps(manifest, indent=2).encode()
                entry = tarfile.TarInfo("BACKUP_MANIFEST.json")
                entry.size, entry.mode, entry.mtime = len(raw), 0o600, 0
                archive.addfile(entry, io.BytesIO(raw))
        with temporary.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    path = out / (checksum + ".tar.gz")
    temporary.replace(path)
    path.chmod(0o600)
    return path, checksum, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    path, checksum, manifest = build(ROOT / "tmp/r7-20261117/backups")
    receipt = {"local_path": str(path), "sha256": checksum, "files": len(manifest["files"]),
               "bytes": path.stat().st_size, "remote_verified": False}
    if args.upload:
        import modal
        volume = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
        remote = PREFIX + "/" + path.name
        with volume.batch_upload() as batch:
            batch.put_file(path, remote)
        received = hashlib.sha256()
        for block in volume.read_file(remote):
            received.update(block)
        if received.hexdigest() != checksum:
            raise ValueError("Remote backup read-back checksum differs")
        receipt.update(volume="ghana-health-understand-train", remote_path=remote, remote_verified=True)
    target = path.with_suffix(".receipt.json")
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    target.chmod(0o600)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
