"""Explicit private research snapshots with complete remote checksum readback."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

from training_data_inputs import require, save, sha

VOLUME = "ghana-health-understand-train"
SECRET = re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|\b(?:hf_|ghp_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{20,}")


def freeze(items, out):
    require(not out.exists(), "Use a fresh snapshot directory")
    payload = {}
    for item in items:
        label, location = item.split("=", 1)
        require(re.fullmatch(r"[a-z][a-z0-9_-]*", label), "Invalid archive label")
        root = Path(location).absolute()
        require(root.exists() and not root.is_symlink(), "Missing or linked research input")
        paths = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in paths:
            require(not path.is_symlink(), "Linked research input")
            if "__pycache__" in path.parts:
                continue
            if not path.is_file():
                continue
            require(path.suffix in {".py", ".md", ".json", ".jsonl", ".jinja"}, "Unsupported research file: " + str(path))
            require(not any(p.startswith(".env") or p == "__pycache__" for p in path.parts), "Forbidden input")
            name = label + "/" + (path.relative_to(root).as_posix() if root.is_dir() else path.name)
            require(name not in payload, "Duplicate snapshot member")
            raw = path.read_bytes()
            require(not SECRET.search(raw) and not raw.startswith(b"SQLite format"), "Potential secret or user store")
            payload[name] = raw
    require(payload and sum(map(len, payload.values())) <= 256 * 1024**2, "Empty or oversized snapshot")
    out.mkdir(parents=True)
    manifest = {"version": "r4-private-snapshot-v1", "private": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Explicit research artifacts; no user chat stores, credentials or base weights",
        "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for name, raw in payload.items()}}
    save(out / "manifest.json", manifest)
    payload["manifest.json"] = (out / "manifest.json").read_bytes()
    with tarfile.open(out / "research.tar.gz", "w:gz") as archive:
        for name, raw in sorted(payload.items()):
            info = tarfile.TarInfo(name)
            info.mode, info.size, info.mtime = 0o600, len(raw), 0
            archive.addfile(info, io.BytesIO(raw))
    verify(out)
    return {"directory": str(out), "files": len(manifest["files"]), "archive_sha256": sha(out / "research.tar.gz")}


def verify(out):
    manifest = json.loads((out / "manifest.json").read_text())
    require(manifest["version"] == "r4-private-snapshot-v1" and manifest["private"] is True, "Wrong snapshot")
    expected = {**manifest["files"], "manifest.json": {"bytes": (out / "manifest.json").stat().st_size, "sha256": sha(out / "manifest.json")}}
    seen = set()
    with tarfile.open(out / "research.tar.gz", "r:gz") as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            require(entry.isfile() and not name.is_absolute() and ".." not in name.parts
                    and entry.name in expected and entry.name not in seen, "Unexpected archive member")
            require(entry.size == expected[entry.name]["bytes"], "Wrong member size")
            raw = archive.extractfile(entry).read()
            require(hashlib.sha256(raw).hexdigest() == expected[entry.name]["sha256"], "Archive checksum differs")
            seen.add(entry.name)
    require(seen == set(expected), "Incomplete snapshot")
    return manifest


def upload(out, profile):
    verify(out)
    os.environ["MODAL_PROFILE"] = profile
    import modal
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    prefix = "/research-training-r4/" + sha(out / "manifest.json")
    expected = {n: sha(out / n) for n in ("manifest.json", "research.tar.gz")}
    def remote_hash(name):
        h = hashlib.sha256()
        for chunk in volume.read_file(prefix + "/" + name):
            h.update(chunk)
        return h.hexdigest()
    missing = []
    for name, checksum in expected.items():
        try:
            require(remote_hash(name) == checksum, "Conflicting remote snapshot")
        except (FileNotFoundError, modal.exception.NotFoundError):
            missing.append(name)
    if missing:
        with volume.batch_upload(force=False) as batch:
            for name in missing:
                batch.put_file(out / name, prefix + "/" + name)
    require(all(remote_hash(name) == checksum for name, checksum in expected.items()), "Remote checksum differs")
    receipt = {"volume": VOLUME, "prefix": prefix, "profile": profile, "private": True,
        "files": expected, "full_remote_readback_verified": True, "verified_at": datetime.now(timezone.utc).isoformat()}
    save(out / "storage.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "upload"))
    parser.add_argument("--item", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", default="createdliving1000")
    parser.add_argument("--authorize-upload", action="store_true")
    args = parser.parse_args()
    if args.action == "upload":
        require(args.authorize_upload, "Explicit upload authorization required")
    result = freeze(args.item, args.out) if args.action == "freeze" else upload(args.out, args.profile) if args.action == "upload" else verify(args.out)
    print(json.dumps(result, indent=2))
