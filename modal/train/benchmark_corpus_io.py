"""Bounded CPU-only test of cached public teacher checkpoint I/O."""
from pathlib import Path
import modal
from benchmark_stronger_response import cache, artifacts, cpu_image

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
REVISION = "e156cb4efae43fbee1a1ab073f946a1377e6b969"
app = modal.App("ghana-corpus-teacher-io")
if modal.is_local():
    cpu_image = cpu_image.add_local_file(str(Path(__file__).with_name("benchmark_stronger_response.py")), "/root/benchmark_stronger_response.py")


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=600, retries=0,
              max_containers=1, volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def measure():
    import hashlib
    import json
    import shutil
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    index = json.loads((root / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    total = sum((root / name).stat().st_size for name in files)

    def transfer(source, target=None):
        started = time.monotonic()
        digest, count = hashlib.sha256(), 0
        writer = target.open("wb") if target else None
        try:
            with source.open("rb") as handle:
                while chunk := handle.read(16 * 1024 * 1024):
                    digest.update(chunk); count += len(chunk)
                    if writer: writer.write(chunk)
        finally:
            if writer: writer.close()
        checksum = digest.hexdigest()
        expected = source.resolve().name
        if len(expected) == 64 and expected != checksum:
            raise ValueError("Cached checkpoint hash mismatch")
        return {"file": source.name, "bytes": count, "sha256": checksum, "seconds": time.monotonic() - started}

    with tempfile.TemporaryDirectory(prefix="teacher-io-") as folder:
        destination = Path(folder)
        if shutil.disk_usage(destination).free < sum((root / f).stat().st_size for f in files[:4]) * 1.1:
            raise ValueError("Insufficient ephemeral staging capacity")
        direct = transfer(root / files[0])
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            copied = list(pool.map(lambda name: transfer(root / name, destination / name), files[:4]))
        staging_seconds = time.monotonic() - started
        local = transfer(destination / files[0])
        if local["sha256"] != copied[0]["sha256"]: raise ValueError("Staged file differs")
    report = {"model": MODEL, "revision": REVISION, "gpu_used": False,
              "direct_volume_read": direct, "concurrent_staged_files": copied,
              "staging_seconds": staging_seconds, "local_read": local,
              "checkpoint_bytes": total, "estimated_full_stage_seconds":
                  total * staging_seconds / sum(r["bytes"] for r in copied),
              "estimate_note": "Four-shard CPU measurement, not a GPU loading measurement or guarantee."}
    artifacts.reload()
    target = Path("/artifacts/corpus-teacher-io")
    target.mkdir(exist_ok=True)
    (target / f"{time.time_ns()}.json").write_text(json.dumps(report, indent=2))
    artifacts.commit()
    return report
