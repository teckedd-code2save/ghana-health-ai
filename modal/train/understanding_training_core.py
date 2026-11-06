"""Single-device finite-schedule adapter training, with full-state boundary resume.

Attach LoRA, load/verify frozen weights, seed RNG and configure CUDA's deterministic
workspace before calling. Put ``scripts`` on sys.path for response_resume_state.
Rows are unbatched integer sequences; the caller owns selection and tokenization.
There is no packing, padding, truncation, shuffle, autocast or evaluation here.

Each call writes a new attempt directory under output. Resume is explicit, from
the returned checkpoint path, even when reusing output. stop_after_updates is an
absolute completed-update limit; max_seconds is per-call and checked at update
boundaries, so one complete update/checkpoint can exceed it. commit() takes no
arguments and durably flushes checkpoints and final receipts. A failing call raises, writes a
failed report, and never publishes its COMPLETE.json. Discard its in-memory model
and resume the last published checkpoint, since partial gradients are not saved.

The helper's run identity is derived from the caller's run hash, encoded rows,
schedule, core/helper code and optimizer recipe. Frozen base identity remains the
caller's responsibility. Checkpoint files contain adapters/buffers, not a base
clone. Imports do not initialize Torch, mutate RNG or modify sys.path.
"""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import response_resume_state as checkpoint_state


VERSION = 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path, value):
    data = _json(value)
    fd, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _digest(value):
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), "Expected lowercase SHA-256 identity")
    return value


def _inputs(rows, schedule):
    _require(type(rows) is list and bool(rows), "rows must be a nonempty list")
    encoded, counts, hasher = [], [], hashlib.sha256()
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"row {index}: expected encoded dictionary")
        copied = {}
        for field in ("input_ids", "attention_mask", "labels"):
            values = row.get(field)
            _require(type(values) in (list, tuple) and all(type(v) is int for v in values),
                     f"row {index}: {field} must be an integer sequence")
            copied[field] = list(values)
        ids, mask, labels = (copied[k] for k in ("input_ids", "attention_mask", "labels"))
        _require(len(ids) >= 2 and len(ids) == len(mask) == len(labels),
                 f"row {index}: mismatched or too short sequences")
        _require(all(v >= 0 for v in ids) and all(v in (0, 1) for v in mask) and
                 all(v == -100 or v >= 0 for v in labels), f"row {index}: invalid token/mask")
        _require(all(label == -100 or mask[i] == 1 for i, label in enumerate(labels)),
                 f"row {index}: supervised padding")
        _require(all(labels[i] == -100 or mask[i - 1] == 1 for i in range(1, len(ids))),
                 f"row {index}: target after masked context")
        count = sum(label != -100 for label in labels[1:])
        _require(count > 0, f"row {index}: no effective causal target tokens")
        encoded.append(copied)
        counts.append(count)
        hasher.update(_json(copied) + b"\n")
    _require(type(schedule) is list and bool(schedule) and
             all(type(i) is int and 0 <= i < len(rows) for i in schedule),
             "schedule must be a nonempty list of valid row indices")
    order = list(schedule)
    return encoded, order, counts, hasher.hexdigest(), checkpoint_state.identity_hash({"indices": order})


def _lr_scale(step, total, warmup):
    if step >= total:
        return 0.0
    if warmup and step < warmup:
        return (step + 1) / warmup
    return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@contextmanager
def _deterministic():
    import torch
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _commit(callback):
    if callback is None:
        return
    import numpy as np
    import torch
    # Storage/logging must not change the training RNG stream.
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        callback()
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def train_schedule(model, rows, schedule, *, output, run_sha256, data_sha256,
                   learning_rate=5e-5, grad_accum=16, warmup_fraction=.05,
                   checkpoint_every=25, resume=None, stop_after_updates=None,
                   max_seconds=None, commit=None):
    """Train an immutable finite order and return a JSON-compatible progress report.

    Optimization: AdamW (.9/.95, eps 1e-8, weight decay .01), gradient norm cap
    1.0, linear warmup then cosine decay, one row per microbatch. The scheduler
    always uses ceil(len(schedule)/grad_accum), never the interruption limit.
    Loss is FP32 summed shifted causal cross entropy / actual window targets.
    An initial checkpoint allows recovery even if the first update fails.
    """
    started = time.monotonic()
    import torch
    import torch.nn.functional as functional
    _digest(run_sha256)
    _digest(data_sha256)
    _require(type(grad_accum) is int and grad_accum > 0, "grad_accum must be positive")
    _require(type(checkpoint_every) is int and checkpoint_every > 0, "checkpoint_every must be positive")
    _require(type(learning_rate) in (int, float) and math.isfinite(learning_rate) and learning_rate > 0,
             "learning_rate must be finite and positive")
    _require(type(warmup_fraction) in (int, float) and math.isfinite(warmup_fraction) and
             0 <= warmup_fraction <= 1, "warmup_fraction must lie in [0, 1]")
    _require(stop_after_updates is None or (type(stop_after_updates) is int and stop_after_updates >= 0),
             "stop_after_updates must be a nonnegative absolute limit")
    _require(max_seconds is None or (type(max_seconds) in (int, float) and
             math.isfinite(max_seconds) and max_seconds >= 0), "max_seconds must be finite and nonnegative")
    _require(commit is None or callable(commit), "commit must be callable")
    encoded, order, targets, rows_sha, schedule_sha = _inputs(rows, schedule)
    total_updates = math.ceil(len(order) / grad_accum)
    warmup = math.ceil(total_updates * warmup_fraction)
    named = list(model.named_parameters())
    trainable = [(name, p) for name, p in named if p.requires_grad]
    _require(bool(trainable), "No trainable adapter parameters")
    tensors = [p for _, p in named] + list(model.buffers())
    devices = {p.device for p in tensors}
    _require(len(devices) == 1 and next(iter(devices)).type in ("cpu", "cuda"),
             "Training requires one CPU or CUDA device, with no offloaded/meta tensors")
    device = next(iter(devices))
    _require(all(p.grad is None for _, p in named), "Start/resume requires cleared gradients")
    _require(all(torch.isfinite(p).all().item() for _, p in trainable), "Nonfinite adapter parameters")
    parameters = [p for _, p in trainable]
    contract = {"version": VERSION, "run_sha256": run_sha256, "data_sha256": data_sha256,
                "encoded_rows_sha256": rows_sha, "schedule_sha256": schedule_sha,
                "core_sha256": _sha(__file__), "resume_helper_sha256": _sha(checkpoint_state.__file__),
                "rows": len(encoded), "microbatches": len(order), "total_updates": total_updates,
                "grad_accum": grad_accum, "learning_rate": float(learning_rate),
                "warmup_updates": warmup, "warmup_fraction": float(warmup_fraction),
                "optimizer": "AdamW", "betas": [.9, .95], "eps": 1e-8, "weight_decay": .01,
                "max_grad_norm": 1.0, "scheduler": "linear-warmup-cosine-v1",
                "loss": "sum-shifted-fp32-cross-entropy/window-effective-targets"}
    bound_sha = checkpoint_state.identity_hash(contract)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=(.9, .95), eps=1e-8,
                                 weight_decay=.01, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_scale(step, total_updates, warmup))
    position = {"completed_updates": 0, "completed_microbatches": 0, "microbatches_in_update": 0,
                "dataloader": {"num_workers": 0, "next_index": 0},
                "sampler": {"epoch": 0, "schedule_sha256": schedule_sha, "total_updates": total_updates},
                "exposure": {"row_visits": {}, "input_tokens": 0, "effective_target_tokens": 0,
                             "loss_sum": 0.0}}
    sampler = dict(position["sampler"])

    def restore_cursor(saved):
        nonlocal position
        updates, cursor = saved["completed_updates"], saved["completed_microbatches"]
        _require(0 <= updates <= total_updates and cursor == min(updates * grad_accum, len(order)) and
                 saved["dataloader"] == {"num_workers": 0, "next_index": cursor} and
                 saved["sampler"] == sampler, "Checkpoint schedule/cursor mismatch")
        exposure = saved["exposure"]
        _require(set(exposure) == {"row_visits", "input_tokens", "effective_target_tokens", "loss_sum"},
                 "Incomplete exposure accounting")
        _require(exposure["row_visits"] == dict(Counter(str(i) for i in order[:cursor])) and
                 exposure["input_tokens"] == sum(len(encoded[i]["input_ids"]) for i in order[:cursor]) and
                 exposure["effective_target_tokens"] == sum(targets[i] for i in order[:cursor]) and
                 type(exposure["loss_sum"]) in (float, int) and math.isfinite(exposure["loss_sum"]) and
                 exposure["loss_sum"] >= 0, "Checkpoint exposure mismatch")
        expected_lr = float(learning_rate) * _lr_scale(updates, total_updates, warmup)
        _require(scheduler.last_epoch == updates and scheduler.get_last_lr() == [expected_lr] and
                 optimizer.param_groups[0]["lr"] == expected_lr, "Checkpoint scheduler/LR mismatch")
        position = saved

    if resume is not None:
        checkpoint_state.restore_checkpoint(
            Path(resume), run_sha256=bound_sha, data_sha256=data_sha256, model=model,
            optimizer=optimizer, scheduler=scheduler, restore_cursor=restore_cursor)
        _require(model.training, "Checkpoint model is not in training mode")
    else:
        model.train()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    try:
        with contract_path.open("xb") as handle:
            handle.write(_json(contract))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _require(json.loads(contract_path.read_bytes()) == contract, "Output contract mismatch")
        _require(resume is not None, "Existing output requires explicit resume or a new output directory")
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=output))
    latest = None
    checkpoint_update = None
    expected_targets = sum(targets[i] for i in order)
    expected_inputs = sum(len(encoded[i]["input_ids"]) for i in order)
    metrics_path = attempt / "steps.jsonl"

    def report(status, reason=None):
        exposure = position["exposure"]
        return {"version": VERSION, "status": status, "state": status, "stop_reason": reason,
                "run_sha256": run_sha256, "data_sha256": data_sha256,
                "checkpoint_run_sha256": bound_sha, "schedule_sha256": schedule_sha,
                "encoded_rows_sha256": rows_sha, "attempt": str(attempt.resolve()),
                "checkpoint": str(latest.resolve()) if latest is not None else None,
                "checkpoint_completed_updates": checkpoint_update,
                "steps_path": str(metrics_path.resolve()), "position": position,
                "planned_updates": total_updates, "planned_microbatches": len(order),
                "planned_effective_target_tokens": expected_targets,
                "planned_input_tokens": expected_inputs,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "trainable_tensors": len(parameters),
                "frozen_parameters": sum(p.numel() for _, p in named if not p.requires_grad),
                "mean_loss": exposure["loss_sum"] / exposure["effective_target_tokens"]
                if exposure["effective_target_tokens"] else None,
                "next_learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.monotonic() - started,
                "completion_verified": status == "completed"}

    def save():
        nonlocal latest, checkpoint_update
        if checkpoint_update != position["completed_updates"]:
            latest = checkpoint_state.save_checkpoint(
                attempt / f"update-{position['completed_updates']:08d}",
                run_sha256=bound_sha, data_sha256=data_sha256, model=model,
                optimizer=optimizer, scheduler=scheduler, position=position)
            checkpoint_update = position["completed_updates"]
        _write_json(attempt / "report.json", report("running"))
        _commit(commit)

    try:
        with _deterministic(), metrics_path.open("xb") as metrics:
            save()
            reason = None
            while position["completed_microbatches"] < len(order):
                if stop_after_updates is not None and position["completed_updates"] >= stop_after_updates:
                    reason = "update_limit"
                    break
                if max_seconds is not None and time.monotonic() - started >= max_seconds:
                    reason = "time_limit"
                    break
                tick = time.monotonic()
                cursor = position["completed_microbatches"]
                window = order[cursor:cursor + grad_accum]
                denominator = sum(targets[i] for i in window)
                loss_sum = 0.0
                lr = optimizer.param_groups[0]["lr"]
                for index in window:
                    row = {key: torch.tensor(value, dtype=torch.long, device=device).unsqueeze(0)
                           for key, value in encoded[index].items()}
                    result = model(input_ids=row["input_ids"], attention_mask=row["attention_mask"], use_cache=False)
                    logits = result["logits"] if isinstance(result, dict) else result.logits
                    _require(logits.ndim == 3 and logits.shape[:2] == row["input_ids"].shape,
                             "Expected full untruncated causal logits")
                    loss = functional.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                                                    row["labels"][:, 1:].reshape(-1),
                                                    ignore_index=-100, reduction="sum")
                    _require(torch.isfinite(loss).item(), "Nonfinite causal loss")
                    (loss / denominator).backward()
                    loss_sum += loss.detach().item()
                    del result, logits, loss, row
                _require(all(p.grad is None for _, p in named if not p.requires_grad),
                         "Frozen parameter received a gradient")
                gradients = [p.grad for p in parameters if p.grad is not None]
                _require(bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients),
                         "Missing or nonfinite adapter gradients")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True, foreach=False)
                optimizer.step()
                _require(all(torch.isfinite(p).all().item() for p in parameters), "Nonfinite updated parameters")
                _require(all(torch.isfinite(t).all().item() for state in optimizer.state.values()
                             for t in state.values() if isinstance(t, torch.Tensor)), "Nonfinite optimizer state")
                scheduler.step()
                model.zero_grad(set_to_none=True)
                position["completed_updates"] += 1
                position["completed_microbatches"] += len(window)
                position["dataloader"]["next_index"] = position["completed_microbatches"]
                exposure = position["exposure"]
                for index in window:
                    key = str(index)
                    exposure["row_visits"][key] = exposure["row_visits"].get(key, 0) + 1
                    exposure["input_tokens"] += len(encoded[index]["input_ids"])
                exposure["effective_target_tokens"] += denominator
                exposure["loss_sum"] += loss_sum
                step = {"update": position["completed_updates"], "start_index": cursor,
                        "next_index": position["completed_microbatches"], "row_indices": window,
                        "microbatches": len(window), "effective_target_tokens": denominator,
                        "input_tokens": sum(len(encoded[i]["input_ids"]) for i in window),
                        "loss_sum": loss_sum, "loss": loss_sum / denominator,
                        "learning_rate": lr, "next_learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm.item(), "gradient_tensors": len(gradients),
                        "seconds": time.monotonic() - tick}
                metrics.write(_json(step) + b"\n")
                metrics.flush()
                os.fsync(metrics.fileno())
                if position["completed_updates"] % checkpoint_every == 0:
                    save()
            save()
            complete = position["completed_microbatches"] == len(order)
            if complete:
                restore_cursor(position)
                _require(position["completed_updates"] == total_updates and
                         position["exposure"]["effective_target_tokens"] == expected_targets and
                         position["exposure"]["input_tokens"] == expected_inputs,
                         "Completion accounting does not match the finite schedule")
            result = report("completed" if complete else "paused", reason)
            _write_json(attempt / "report.json", result)
            if complete:
                _write_json(attempt / "COMPLETE.json", result)
            _commit(commit)
            return result
    except BaseException as error:
        model.zero_grad(set_to_none=True)
        (attempt / "COMPLETE.json").unlink(missing_ok=True)
        failed = report("failed", type(error).__name__)
        failed["error"] = str(error)
        _write_json(attempt / "report.json", failed)
        try:
            _commit(commit)
        except BaseException:
            # A remote commit that raises has an unknown durability outcome.
            # The wrapper must treat it as failed, never as a completed run.
            pass
        raise


def assert_checkpoint_equivalent(left, right):
    """Checksum-load two trusted local checkpoints and require bit-exact state.

    This compares all saved state, including optimizer, scheduler, modes, buffers,
    RNG and exposure, without materializing frozen model weights. Timing/report
    metadata lives outside state.pt. Raises ValueError with the first differing
    field; success returns compact verification metrics for a smoke receipt.
    """
    import torch

    def read(path):
        path = Path(path)
        identity = json.loads((path / "manifest.json").read_bytes())["identity"]
        # Reuse the helper's checksummed, weights-only reader, without a model.
        return checkpoint_state._read_checkpoint(path, identity)

    def equal(a, b, location):
        _require(type(a) is type(b), f"Checkpoint difference at {location}: type")
        if isinstance(a, torch.Tensor):
            _require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b),
                     f"Checkpoint difference at {location}: tensor")
        elif isinstance(a, dict):
            _require(a.keys() == b.keys(), f"Checkpoint difference at {location}: keys")
            for key in a:
                equal(a[key], b[key], f"{location}.{key}")
        elif isinstance(a, (list, tuple)):
            _require(len(a) == len(b), f"Checkpoint difference at {location}: length")
            for index, (aa, bb) in enumerate(zip(a, b, strict=True)):
                equal(aa, bb, f"{location}[{index}]")
        else:
            _require(a == b, f"Checkpoint difference at {location}")

    a, b = read(left), read(right)
    equal(a, b, "state")
    return {"equivalent": True, "checked_sections": sorted(a),
            "completed_updates": a["position"]["completed_updates"],
            "effective_target_tokens": a["position"]["exposure"]["effective_target_tokens"]}
