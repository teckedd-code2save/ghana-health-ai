"""Follow one existing R4 call; never launch or retry training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import run_understanding_r4 as runner


def monitor(receipt, out):
    launch = runner.read(receipt)
    runner.require(launch["state"] == "submitted" and launch["mode"] == "train", "Existing training receipt required")
    os.environ["MODAL_PROFILE"] = launch["profile"]
    import modal
    call = modal.FunctionCall.from_id(launch["call_id"])
    volume = modal.Volume.from_name(runner.VOLUME, create_if_missing=False)
    out.mkdir(parents=True, exist_ok=True)
    deadline, last = time.monotonic() + 4 * 3600, None
    while time.monotonic() < deadline:
        try:
            terminal = call.get(timeout=60)
        except TimeoutError:
            terminal = None
        except Exception as error:
            runner.save(out / "monitor-error.json", {"type": type(error).__name__, "message": str(error)[:1000],
                        "run_id": launch["run_id"], "automatic_retry": False})
            raise
        raw = b"".join(volume.read_file(launch["remote_directory"] + "/consumption.json"))
        consumption = json.loads(raw)
        runner.require(all(consumption[k] == launch[k] for k in ("run_id", "execution_sha256", "data_sha256")), "Consumption belongs to another run")
        cursor = consumption["completed_updates"]
        if cursor != last:
            runner.save(out / f"consumption-{cursor:04d}.json", consumption)
            print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "completed_updates": cursor,
                "row_exposures": consumption["consumed"]["row_exposures"],
                "effective_target_tokens": consumption["consumed"]["effective_target_tokens"]}), flush=True)
            last = cursor
        if terminal is not None:
            runner.require(terminal["run_id"] == launch["run_id"], "Terminal belongs to another run")
            runner.save(out / "terminal.json", terminal)
            print(json.dumps({"terminal": terminal["state"], "run_id": launch["run_id"]}), flush=True)
            return terminal
    raise TimeoutError("Monitor bound reached; inspect the same call, do not submit a duplicate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = monitor(args.receipt, args.out)
    if result["state"] != "completed" or not result.get("training_completed"):
        raise SystemExit("Training did not complete: " + result["state"])
