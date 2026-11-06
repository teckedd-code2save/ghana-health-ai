"""Read-only recovery selection; no Torch, Modal, network calls or clock leases.

The caller retrieves claims[previous_run.name] and positively confirms that its
Modal FunctionCall is terminal. A poll timeout, transport error, or stale status
file is NOT terminal confirmation. This function validates that supplied evidence
is bound to the requested run; it cannot authenticate a fabricated caller record.

Checkpoint contents remain opaque here. We verify publication seals, identities,
byte counts and SHA-256; train_schedule must subsequently validate and restore the
full state. The update number is a candidate cursor, not proof of the saved state.
Only the highest stage/update is eligible. A damaged newest checkpoint is an error,
not permission to silently restart from older state. Unpublished .tmp directories
are ignored. Equal-update attempts must have identical checkpoint state bytes.
"""

import hashlib
import json
import math
from pathlib import Path
import re


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _digest(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256")
    return value


def _file(path):
    _require(not path.is_symlink() and path.is_file(), f"Missing/unsafe file: {path}")
    return path


def _directory(path):
    _require(not path.is_symlink() and path.is_dir(), f"Missing/unsafe directory: {path}")
    return path


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON field in {path}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value in {path}: {value}")

    value = json.loads(_file(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    _require(isinstance(value, dict), f"Expected JSON object in {path}")
    return value


def select_recovery(previous_run, *, runs_root, execution_sha256, data_sha256,
                    stages, grad_accum, claim, terminal_call):
    """Return a checked stage_index/checkpoint candidate for train_schedule resume.

    stages is the ordered pack stage list, each with name and full schedule.
    claim is fetched by run ID from the existing Modal claim dictionary.
    terminal_call is {call_id, state}, where state is a positively confirmed
    completed/failed/timed_out/cancelled call. Polling timeouts are not timed_out.

    Stale training/failed/paused statuses do not determine resumability. A stage
    completed just before budget exhaustion is selected too: invoke train_schedule
    on that same full stage with resume=checkpoint; it restores state and returns
    completed without replay, after which the next stage gets a fresh optimizer.
    """
    _digest(execution_sha256)
    _digest(data_sha256)
    _require(type(grad_accum) is int and grad_accum > 0, "Invalid accumulation window")
    root, previous = Path(runs_root), Path(previous_run)
    _require(root.is_absolute() and root.resolve() == root, "runs_root must be canonical and absolute")
    _directory(root)
    _require(previous.is_absolute() and previous.parent == root and previous.resolve() == previous,
             "Previous run must be a direct canonical child of runs_root")
    _directory(previous)
    _require(re.fullmatch(r"understanding_\d{8}T\d{6}Z", previous.name), "Invalid run directory name")
    _require(isinstance(claim, dict) and claim.get("execution_sha256") == execution_sha256 and
             isinstance(claim.get("call_id"), str) and bool(claim["call_id"]), "Claim identity mismatch")
    _require(isinstance(terminal_call, dict) and terminal_call.get("call_id") == claim["call_id"] and
             terminal_call.get("state") in {"completed", "failed", "timed_out", "cancelled"},
             "Matching positive terminal-call confirmation is required")
    execution_path = _file(previous / "execution.json")
    _require(_sha(execution_path) == execution_sha256, "Execution contract checksum mismatch")
    execution = _read(execution_path)
    _require(execution.get("mode") == "train" and execution.get("data_sha256") == data_sha256,
             "Execution mode/data identity mismatch")
    status = _read(previous / "status.json")
    _require(status.get("mode") == "train" and status.get("run_id") == previous.name and
             status.get("execution_sha256") == execution_sha256 and
             status.get("data_sha256") == data_sha256, "Status identity mismatch")
    _require(isinstance(stages, list) and bool(stages), "Expected ordered stages")
    names, candidates = set(), []
    for stage_index, stage in enumerate(stages):
        _require(isinstance(stage, dict), "Malformed stage")
        name, schedule = stage.get("name"), stage.get("schedule")
        _require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_-]*", name) and name not in names,
                 "Unsafe or repeated stage name")
        names.add(name)
        _require(type(schedule) is list and bool(schedule) and
                 all(type(i) is int and i >= 0 for i in schedule), "Invalid stage schedule")
        stage_dir = previous / name
        if not stage_dir.exists() and not stage_dir.is_symlink():
            continue
        _directory(stage_dir)
        contract = _read(stage_dir / "contract.json")
        total = math.ceil(len(schedule) / grad_accum)
        _require(contract.get("run_sha256") == execution_sha256 and
                 contract.get("data_sha256") == data_sha256 and
                 contract.get("schedule_sha256") == _hash({"indices": schedule}) and
                 contract.get("grad_accum") == grad_accum and contract.get("total_updates") == total and
                 contract.get("microbatches") == len(schedule), "Stage contract identity/schedule mismatch")
        identity = {"run_sha256": _hash(contract), "data_sha256": data_sha256}
        for attempt in stage_dir.iterdir():
            if not attempt.name.startswith("attempt-"):
                continue
            _directory(attempt)
            for checkpoint in attempt.iterdir():
                if not checkpoint.name.startswith("update-"):
                    continue
                _directory(checkpoint)
                match = re.fullmatch(r"update-(\d{8,})", checkpoint.name)
                _require(match is not None, "Malformed checkpoint update directory")
                update = int(match[1])
                _require(0 <= update <= total, "Checkpoint update exceeds finite stage")
                candidates.append((stage_index, update, checkpoint, identity, total))
    _require(bool(candidates), "No published checkpoint candidate; do not restart automatically")
    newest = max((stage, update) for stage, update, *_ in candidates)
    latest = [candidate for candidate in candidates if candidate[:2] == newest]
    checked = []
    for stage_index, update, checkpoint, identity, total in latest:
        _require({p.name for p in checkpoint.iterdir()} == {"state.pt", "manifest.json", "COMMITTED.json"},
                 "Newest checkpoint publication is incomplete; refusing fallback")
        for name in ("state.pt", "manifest.json", "COMMITTED.json"):
            _file(checkpoint / name)
        seal = _read(checkpoint / "COMMITTED.json")
        _require(set(seal) == {"manifest_sha256"} and
                 _digest(seal["manifest_sha256"]) == _sha(checkpoint / "manifest.json"),
                 "Newest checkpoint manifest checksum mismatch; refusing fallback")
        manifest = _read(checkpoint / "manifest.json")
        _require(set(manifest) == {"version", "identity", "state_sha256", "state_bytes"} and
                 type(manifest["version"]) is int and manifest["version"] == 1 and
                 manifest["identity"] == identity, "Newest checkpoint identity/version mismatch")
        _digest(manifest["state_sha256"])
        _require(type(manifest["state_bytes"]) is int and manifest["state_bytes"] > 0 and
                 (checkpoint / "state.pt").stat().st_size == manifest["state_bytes"] and
                 _sha(checkpoint / "state.pt") == manifest["state_sha256"],
                 "Newest checkpoint state checksum/size mismatch; refusing fallback")
        checked.append((str(checkpoint), manifest["state_sha256"], identity["run_sha256"], total))
    _require(len({state_sha for _, state_sha, _, _ in checked}) == 1,
             "Conflicting newest checkpoints at the same update; explicit adjudication required")
    path, state_sha, checkpoint_run_sha, total = min(checked)
    stage_index, update = newest
    return {"stage_index": stage_index, "stage": stages[stage_index]["name"], "checkpoint": path,
            "completed_updates": update, "stage_completed": update == total,
            "run_id": previous.name, "call_id": claim["call_id"],
            "execution_sha256": execution_sha256, "data_sha256": data_sha256,
            "checkpoint_run_sha256": checkpoint_run_sha, "state_sha256": state_sha,
            "checkpoints_seen": len(candidates), "newest_checkpoints_verified": len(checked),
            "checkpoint_checksums_verified": True, "full_state_restore_required": True}
