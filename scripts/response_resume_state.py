"""Local, single-process LoRA resume checkpoints; not wired into any runner.

Integration contract:
* Hash the frozen run contract (base/adapter/tokenizer revisions, code, runtime,
  optimizer recipe, finite exposure schedule) and the verified data manifest.
  These hashes bind identity; this module does not verify frozen base weights.
* Save after optimizer.step(), scheduler.step(), cursor/accounting advancement,
  and model.zero_grad(set_to_none=True). No gradients, AMP scaler, distributed
  state or mid-accumulation resume are supported. Use CPU or initialized CUDA.
* ``position`` contains completed_updates, completed_microbatches,
  microbatches_in_update (always zero), and nonempty dataloader/sampler/exposure
  dictionaries. Dataloader num_workers must be zero. Save the *next unconsumed*
  position, order/epoch, owned generator states and canonical target/token reuse
  counts, not the prefetched iterator position. Values are tensors or primitives.
* Rebuild the verified base, identical trainable parameters, optimizer AND
  scheduler on their final devices before restore. All named trainable parameters
  and buffers are saved; immutable frozen parameters are deliberately omitted.
  Model-specific extra state must be handled before adopting this helper.
* Restore validates first, then loads weights/modes, scheduler, optimizer,
  invokes restore_cursor(position), and restores global RNG LAST. The callback
  must rebuild the iterator and restore its owned generators before returning;
  it must not fetch a training batch. Do not reseed/step/rebuild afterwards.
  If any load/callback raises, discard the partially restored objects.

Checkpoints are immutable directories published by same-filesystem rename after
fsync. A caller must supply a unique destination and serialize training/saving.
Checksums detect damage, not malicious replacement. Remote volume commit/upload,
runner packaging/identity checks, real-weight/GPU smoke tests and recovery-copy
verification remain integration work. Imports do not load Torch or change runtime.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile


VERSION = 1
POSITION_KEYS = {"completed_updates", "completed_microbatches", "microbatches_in_update",
                 "dataloader", "sampler", "exposure"}
STATE_KEYS = {"version", "identity", "runtime", "model", "model_modes", "optimizer",
              "scheduler", "position", "rng"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, keys, label):
    _require(isinstance(value, dict) and set(value) == set(keys),
             f"Incomplete or unexpected {label} fields")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identity_hash(contract):
    """Hash a nonempty JSON-object run contract or data manifest canonically."""
    _require(isinstance(contract, dict) and bool(contract), "Identity contract must be nonempty")
    encoded = _json(contract)
    _require(json.loads(encoded) == contract, "Identity contract must contain JSON values")
    return hashlib.sha256(encoded).hexdigest()


def _identity(run_sha256, data_sha256):
    for value in (run_sha256, data_sha256):
        _require(isinstance(value, str) and len(value) == 64 and
                 all(c in "0123456789abcdef" for c in value), "Expected lowercase SHA-256 identity")
    return {"run_sha256": run_sha256, "data_sha256": data_sha256}


def _snapshot(value):
    """Detach mutable state and restrict serialization to weights-only data."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if type(value) in (str, int, float, bool, bytes, type(None)):
        return value
    if isinstance(value, dict):
        _require(all(type(k) in (str, int) for k in value), "State keys must be strings or integers")
        return {k: _snapshot(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_snapshot(v) for v in value)
    raise ValueError(f"Unsupported checkpoint state type: {type(value).__name__}")


def _model_tensors(model):
    return {"parameters": {n: p for n, p in model.named_parameters() if p.requires_grad},
            "buffers": dict(model.named_buffers())}


def trainable_state_dict(model):
    """Defined model state: named trainable parameters plus ALL named buffers."""
    return _snapshot(_model_tensors(model))


def _position(position):
    _keys(position, POSITION_KEYS, "position")
    for key in ("completed_updates", "completed_microbatches", "microbatches_in_update"):
        _require(type(position[key]) is int and position[key] >= 0, f"Invalid {key}")
    _require(position["microbatches_in_update"] == 0,
             "Only optimizer update boundaries are supported; gradients are not saved")
    _require(position["completed_microbatches"] >= position["completed_updates"],
             "Microbatch count precedes completed updates")
    for key in ("dataloader", "sampler", "exposure"):
        _require(isinstance(position[key], dict) and bool(position[key]), f"Missing {key} cursor")
    _require(type(position["dataloader"].get("num_workers")) is int and
             position["dataloader"]["num_workers"] == 0, "Worker/prefetch state is not supported")


def _boundary(model):
    _require(all(p.grad is None for p in model.parameters()),
             "Clear gradients with set_to_none=True at an optimizer update boundary")


def _runtime(model, optimizer, scheduler):
    import numpy as np
    import torch
    _require(getattr(scheduler, "optimizer", None) is optimizer,
             "Construct the scheduler with the supplied optimizer before resume")
    named = dict(model.named_parameters())
    names = {id(p): n for n, p in named.items() if p.requires_grad}
    groups = [[names.get(id(p)) for p in group["params"]] for group in optimizer.param_groups]
    flat = [name for group in groups for name in group]
    _require(bool(names) and None not in flat and len(flat) == len(set(flat)) and
             set(flat) == set(names.values()), "Optimizer must cover each trainable parameter exactly once")
    tensors = {**named, **dict(model.named_buffers())}
    _require(all(t.device.type in ("cpu", "cuda") for t in tensors.values()),
             "Only CPU/CUDA resume is supported")
    return {"python": list(sys.version_info[:3]), "torch": str(torch.__version__), "numpy": np.__version__,
            "optimizer": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
            "scheduler": type(scheduler).__module__ + "." + type(scheduler).__qualname__,
            "parameter_groups": groups,
            "tensors": {n: [list(t.shape), str(t.dtype), str(t.device), bool(t.requires_grad)]
                        for n, t in tensors.items()}}


def _rng_state():
    import numpy as np
    import torch
    algorithm, keys, index, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (algorithm, keys.tolist(), index, gaussian, cached),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _numpy_rng(state):
    import numpy as np
    _require(isinstance(state, tuple) and len(state) == 5, "Incomplete NumPy RNG state")
    algorithm, keys, index, gaussian, cached = state
    _require(algorithm == "MT19937" and isinstance(keys, list) and len(keys) == 624 and
             all(type(k) is int and 0 <= k < 2**32 for k in keys) and
             type(index) is int and 0 <= index <= 624 and gaussian in (0, 1), "Invalid NumPy RNG state")
    return algorithm, np.array(keys, dtype=np.uint32), index, gaussian, cached


def _validate_rng(rng):
    import numpy as np
    import torch
    _keys(rng, {"python", "numpy", "torch_cpu", "torch_cuda"}, "RNG")
    try:
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(_numpy_rng(rng["numpy"]))
        torch.Generator(device="cpu").set_state(rng["torch_cpu"])
        cuda = rng["torch_cuda"]
        count = torch.cuda.device_count() if torch.cuda.is_initialized() else 0
        _require(isinstance(cuda, list) and len(cuda) == count, "CUDA RNG device count mismatch")
        if count:
            current = torch.cuda.get_rng_state_all()
            _require(all(isinstance(s, torch.Tensor) and s.dtype == torch.uint8 and
                         s.device.type == "cpu" and s.shape == old.shape
                         for s, old in zip(cuda, current, strict=True)), "Invalid CUDA RNG state")
    except (TypeError, RuntimeError, OverflowError) as error:
        raise ValueError("Invalid RNG state") from error


def _restore_rng(rng):
    import numpy as np
    import torch
    random.setstate(rng["python"])
    np.random.set_state(_numpy_rng(rng["numpy"]))
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def _validate_state(state, identity, model, optimizer, scheduler):
    import torch
    _keys(state, STATE_KEYS, "checkpoint")
    _require(state["version"] == VERSION, "Unsupported checkpoint version")
    _require(state["identity"] == identity, "Run/data identity mismatch")
    _require(state["runtime"] == _runtime(model, optimizer, scheduler),
             "Runtime, model or optimizer parameter-order mismatch")
    _position(state["position"])
    _validate_rng(state["rng"])
    current = _model_tensors(model)
    _keys(state["model"], current, "model")
    for kind, tensors in current.items():
        _keys(state["model"][kind], tensors, kind)
        for name, tensor in tensors.items():
            saved = state["model"][kind][name]
            _require(isinstance(saved, torch.Tensor) and saved.shape == tensor.shape and
                     saved.dtype == tensor.dtype, f"Model tensor mismatch: {name}")
    _keys(state["model_modes"], dict(model.named_modules()), "model modes")
    _require(all(type(v) is bool for v in state["model_modes"].values()), "Invalid model modes")
    _keys(state["scheduler"], scheduler.state_dict(), "scheduler")
    _keys(state["optimizer"], {"state", "param_groups"}, "optimizer")
    groups = state["optimizer"]["param_groups"]
    expected = optimizer.state_dict()["param_groups"]
    _require(isinstance(groups, list) and len(groups) == len(expected), "Incomplete optimizer groups")
    for group, target in zip(groups, expected, strict=True):
        _keys(group, target, "optimizer group")
        _require(group["params"] == target["params"], "Optimizer parameter IDs mismatch")
    ids = {p for group in groups for p in group["params"]}
    slots = state["optimizer"]["state"]
    _require(isinstance(slots, dict) and set(slots) <= ids and
             all(isinstance(v, dict) for v in slots.values()), "Invalid optimizer slots")


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write(path, data):
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_checkpoint(path, *, run_sha256, data_sha256, model, optimizer, scheduler, position):
    """Publish one immutable full-state checkpoint; never overwrite a prior one."""
    import torch
    identity = _identity(run_sha256, data_sha256)
    _position(position)
    _boundary(model)
    state = _snapshot({"version": VERSION, "identity": identity,
                       "runtime": _runtime(model, optimizer, scheduler),
                       "model": _model_tensors(model),
                       "model_modes": {n: m.training for n, m in model.named_modules()},
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "position": position, "rng": _rng_state()})
    _validate_state(state, identity, model, optimizer, scheduler)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + path.name + ".tmp-", dir=path.parent))
    try:
        with (temporary / "state.pt").open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {"version": VERSION, "identity": identity,
                    "state_sha256": _sha(temporary / "state.pt"),
                    "state_bytes": (temporary / "state.pt").stat().st_size}
        _write(temporary / "manifest.json", _json(manifest))
        _write(temporary / "COMMITTED.json", _json({"manifest_sha256": _sha(temporary / "manifest.json")}))
        _fsync_directory(temporary)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        temporary.rename(path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return path


def _read_checkpoint(path, identity):
    import torch
    path = Path(path)
    try:
        _require(not path.is_symlink() and path.is_dir(), "Missing checkpoint directory")
        _require({p.name for p in path.iterdir()} == {"state.pt", "manifest.json", "COMMITTED.json"},
                 "Incomplete checkpoint publication")
        _require(all(not p.is_symlink() and p.is_file() for p in path.iterdir()), "Invalid checkpoint files")
        seal = json.loads((path / "COMMITTED.json").read_bytes())
        _keys(seal, {"manifest_sha256"}, "commit seal")
        manifest_bytes = (path / "manifest.json").read_bytes()
        _require(hashlib.sha256(manifest_bytes).hexdigest() == seal["manifest_sha256"],
                 "Manifest checksum mismatch")
        manifest = json.loads(manifest_bytes)
        _keys(manifest, {"version", "identity", "state_sha256", "state_bytes"}, "manifest")
        _require(manifest["version"] == VERSION, "Unsupported checkpoint version")
        _require(manifest["identity"] == identity, "Run/data identity mismatch")
        with (path / "state.pt").open("rb") as handle:
            _require(os.fstat(handle.fileno()).st_size == manifest["state_bytes"], "State size mismatch")
            _require(hashlib.file_digest(handle, "sha256").hexdigest() == manifest["state_sha256"],
                     "State checksum mismatch")
            handle.seek(0)
            return torch.load(handle, map_location="cpu", weights_only=True)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Corrupted or incomplete checkpoint") from error


def restore_checkpoint(path, *, run_sha256, data_sha256, model, optimizer, scheduler, restore_cursor):
    """Validate, restore weights -> scheduler -> optimizer -> cursor -> RNG.

    The required callback receives the saved position and must install it, not
    replay/skip already-consumed examples. The returned position is a separate copy.
    """
    import torch
    _require(callable(restore_cursor), "A restore_cursor callback is required")
    identity = _identity(run_sha256, data_sha256)
    state = _read_checkpoint(path, identity)
    _validate_state(state, identity, model, optimizer, scheduler)
    _boundary(model)
    with torch.no_grad():
        for kind, tensors in _model_tensors(model).items():
            for name, tensor in tensors.items():
                tensor.copy_(state["model"][kind][name])
    for name, module in model.named_modules():
        module.training = state["model_modes"][name]
    scheduler.load_state_dict(state["scheduler"])
    optimizer.load_state_dict(state["optimizer"])
    restore_cursor(_snapshot(state["position"]))
    _restore_rng(state["rng"])
    return _snapshot(state["position"])
