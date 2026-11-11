"""Private, single-GPU, one-chunk IFEval worker. No deployment or automatic chaining."""
import importlib.util
import os
from pathlib import Path
import sys

import modal

PACKAGE = Path(os.environ.get("GHA_R5_IFEVAL_PACKAGE", "/execution")) if modal.is_local() else Path("/execution")
sys.path.insert(0, str(PACKAGE / "code"))
path = PACKAGE / "code/run_understanding_r5_ifeval.py"
spec = importlib.util.spec_from_file_location("_bound_r5_ifeval_runner", path)
runner = importlib.util.module_from_spec(spec)
exec(compile(path.read_bytes(), str(path), "exec"), runner.__dict__)
contract = runner.read(PACKAGE / "execution.json")

app = modal.App("ghana-understanding-ifeval-r5-private")
artifacts = modal.Volume.from_name(runner.VOLUME, create_if_missing=False)
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
claims = modal.Dict.from_name("ghana-r5-ifeval-attempts-v1", create_if_missing=True)
ledger = modal.Dict.from_name(runner.LEDGER, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    *[name + "==" + version for name, version in contract["runtime"].items()]
).env({"PYTHONPATH": "/execution/code:/execution/inference/code", "PYTHONDONTWRITEBYTECODE": "1",
       "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
if modal.is_local():
    runner.verify(PACKAGE, runner.sha(PACKAGE / "execution.json"))
    image = image.add_local_dir(str(PACKAGE), "/execution")


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=7200, retries=0,
              max_containers=1, min_containers=0, single_use_containers=True,
              volumes={"/artifacts": artifacts, "/cache": cache.read_only()}, include_source=False)
def evaluate(run_id: str, checksum: str, attempt: int, chunk_slots: int, lease: int, owner: str):
    import importlib.metadata
    import re
    import shutil
    import time

    started = time.monotonic()
    runner.require(re.fullmatch(r"r5_ifeval_\d{8}T\d{6}Z_[a-f0-9]{8}", run_id), "Invalid IFEval run ID")
    runner.require(type(chunk_slots) is int and 1 <= chunk_slots <= runner.LIMITS["maximum_chunk_slots"], "Invalid chunk")
    claim = ledger.get("lease:" + str(lease))
    runner.require(claim is not None and claim["owner"] == owner and claim["run_id"] == run_id
                   and claim["execution_sha256"] == checksum and claim["attempt"] == attempt, "Unbound global GPU lease")
    runner.bind_call(ledger, lease, owner, modal.current_function_call_id())
    execution, manifest, module, rows = runner.verify(PACKAGE, checksum)
    core, _ = runner.common.activate(PACKAGE)
    claims.put(run_id + ":binding", checksum, skip_if_exists=True)
    runner.require(claims.get(run_id + ":binding") == checksum, "Run already bound to a different execution")
    core.claim_attempt(claims, run_id, attempt, modal.current_function_call_id(), runner.common.terminal)
    artifacts.reload()
    cache.reload()
    folder = Path("/artifacts" + runner.PREFIX) / run_id
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    bound = runner.binding(execution, checksum, run_id)
    journal = core.Journal(folder / "journal", bound, artifacts.commit)

    def status(state, **extra):
        value = {**bound, "state": state, "attempt": attempt, "call_id": modal.current_function_call_id(),
                 "lease": lease, "expected_slots": 1623, "chunk_slots": chunk_slots,
                 "elapsed_seconds": time.monotonic() - started, "training": False, "accuracy": None, **extra}
        runner.save(folder / "status.json", value)
        artifacts.commit()
        return value

    status("loading")
    try:
        snapshots = {"execution.json": checksum, "context-preflight.json": execution["files"]["context-preflight.json"],
                     **{n: h for n, h in execution["files"].items() if n.startswith("provenance/")}}
        for name, expected in snapshots.items():
            destination = folder / name
            if destination.exists():
                runner.require(runner.sha(destination) == expected, "Saved execution/provenance differs")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PACKAGE / name, destination)
        consumption = Path(manifest["variants"]["r5_response"]["directory"]).parent / "consumption.json"
        runner.require(runner.sha(consumption) == runner.read(PACKAGE / "provenance/summary.json")["consumption_sha256"],
                       "Mounted consumption receipt differs")
        engine = module.Runtime(PACKAGE / "inference", execution["inference_sha256"])
        runtime = runner.generation.Runtime(engine, core)
        identities = {v: runtime.identity(v) for v in runner.VARIANTS}
        actual_versions = {name: importlib.metadata.version(name) for name in execution["runtime"]}
        runner.require(actual_versions == execution["runtime"], "Installed GPU runtime differs")
        proof = {"binding": bound, "attempt": attempt, "call_id": modal.current_function_call_id(), "lease": lease,
                 "runtime": actual_versions, "loading": engine.loading, "identities": identities,
                 "base_bytes_verified": True, "adapter_bytes_verified": True, "kernel_binding_verified": True,
                 "adapter_tensors": "Verified by pinned Runtime.select before first operation per adapter"}
        proof_path = folder / "runtime" / ("attempt-" + str(attempt) + ".json")
        runner.require(not proof_path.exists(), "Runtime attempt already recorded")
        runner.save(proof_path, proof)
        artifacts.commit()
        for slot in runner.slots(rows):
            record = journal.get(slot["key"])
            if record is not None:
                previous = folder / "runtime" / ("attempt-" + str(record["attempt"]) + ".json")
                runner.require(runner.sha(previous) == record["runtime_proof_sha256"]
                               and runner.read(previous)["binding"] == bound
                               and runner.read(previous)["identities"] == identities, "Prior runtime proof differs")
        status("generating")
        result = runner.evaluate(runtime, core, rows, journal, attempt, runner.sha(proof_path), execution["generation_policy"],
                                 started + execution["limits"]["soft_seconds"], chunk_slots)
    except Exception as error:
        result = {"state": "failed", "error_type": type(error).__name__, "error": str(error)[:2000]}
    try:
        summary = runner.report(folder, rows, journal)
        runner.source.private(folder)
        return status(**result, finalized_slots=summary["finalized_slots"])
    except Exception as error:
        return status("failed_reporting", prior_state=result["state"], error_type=type(error).__name__, error=str(error)[:2000])
