#!/usr/bin/env python3
"""Resume the pinned public foundation locally with aria2; never starts compute."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess

from prepare_r7_local_mlx import BASE_SHA, MODEL, REVISION, require, sha

SIZE = 17907663008


def command(binary, staging):
    return [binary, "--no-conf=true", "--no-netrc=true", "--continue=true",
            "--always-resume=true", "--allow-overwrite=false", "--auto-file-renaming=false",
            "--file-allocation=none", "--split=8", "--max-connection-per-server=8",
            "--min-split-size=32M", "--max-tries=5", "--retry-wait=3",
            "--connect-timeout=20", "--timeout=60", "--auto-save-interval=15",
            "--enable-rpc=false", "--follow-torrent=false", "--follow-metalink=false",
            "--enable-color=false", "--summary-interval=30", "--console-log-level=warn",
            "--download-result=hide", "--checksum=sha-256=" + BASE_SHA,
            "--dir=" + str(staging), "--out=model.safetensors.part",
            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors"]


def safe_line(line):
    return re.sub(r"https?://\S+", "[transfer URL omitted]", line).strip()


def run(root, binary):
    root = root.resolve()
    require(shutil.which(binary), "aria2c is not installed")
    base = root / "afrique-base-bf16"
    staging = root / "base-transfer-aria2"
    final = base / "model.safetensors"
    with (root / "download-base.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final.exists():
            require(final.stat().st_size == SIZE and sha(final) == BASE_SHA, "Wrong completed foundation")
            print(json.dumps({"status": "already_verified", "sha256": BASE_SHA}), flush=True)
            return
        staging.mkdir(mode=0o700, exist_ok=True)
        partial = staging / "model.safetensors.part"
        existing = partial
        if not partial.exists():
            previous = list((base / ".cache/huggingface/download").glob(f"*.{BASE_SHA}.incomplete"))
            require(len(previous) == 1 and 0 < previous[0].stat().st_size < SIZE,
                    "Expected exactly one sequential HF partial download")
            existing = previous[0]
        remaining_allocation = max(0, SIZE - existing.stat().st_blocks * 512)
        require(shutil.disk_usage(root).free > remaining_allocation + 8 * 1024**3,
                "Insufficient local disk headroom")
        if existing != partial:
            existing.rename(partial)
        state = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
                 "status": "downloading", "transport": "aria2", "paid_compute": False,
                 "revision": REVISION, "expected_sha256": BASE_SHA, "expected_bytes": SIZE,
                 "staging": str(staging), "resume_control": str(partial) + ".aria2"}
        destination = root / "download-base-status.json"

        def save():
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(destination)

        save()
        child = None

        def stop(signum, frame):
            if child and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            state.update(status="stopped_incomplete", stopped_signal=signum)
            save()
            raise SystemExit(128 + signum)

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, stop)
        signal.alarm(6 * 3600)
        try:
            child = subprocess.Popen(command(binary, staging), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
            state["transfer_pid"] = child.pid
            save()
            with (root / "download-base-aria2.log").open("a") as log:
                for line in child.stdout:
                    clean = safe_line(line)
                    if not clean:
                        continue
                    log.write(clean + "\n")
                    log.flush()
                    if "CN:" in clean:
                        state["progress"] = clean
                        state["updated_at"] = datetime.now(timezone.utc).isoformat()
                        save()
                    if "CN:" not in clean or "Download Progress Summary" in clean:
                        print(clean, flush=True)
            code = child.wait()
            require(code == 0, f"Download stopped with exit code {code}; partial files retained")
            state["status"] = "verifying_hash"
            save()
            require(partial.stat().st_size == SIZE and sha(partial) == BASE_SHA, "Foundation hash mismatch")
            require(not final.exists(), "Completed target appeared during download")
            partial.rename(final)
            state.update(status="downloaded_and_hash_verified", sha256=BASE_SHA, path=str(final))
        except Exception as error:
            state.update(status="stopped_incomplete", error_type=type(error).__name__)
            raise
        finally:
            signal.alarm(0)
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            save()
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aria2", default="/opt/homebrew/bin/aria2c")
    args = parser.parse_args()
    try:
        run(args.root, args.aria2)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None
