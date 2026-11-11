"""Private bounded R5 evaluator for separate development/fresh and raw/guarded runs."""
import os
import importlib.util
from pathlib import Path
import sys

import modal

PACKAGE = Path(os.environ.get("GHA_R5_EVALUATION_PACKAGE", "/execution")) if modal.is_local() else Path("/execution")
sys.path.insert(0, str(PACKAGE / "code"))
_path = PACKAGE / "code/run_understanding_r5_evaluation.py"
_spec = importlib.util.spec_from_file_location("_r5_evaluation_bound_runner", _path)
runner = importlib.util.module_from_spec(_spec)
exec(compile(_path.read_bytes(), str(_path), "exec"), runner.__dict__)

contract = runner.read(PACKAGE / "execution.json")
app = modal.App("ghana-understanding-evaluation-r5-private")
artifacts = modal.Volume.from_name(runner.VOLUME, create_if_missing=False)
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
claims = modal.Dict.from_name("ghana-understanding-evaluation-r5-invocations", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    *[name + "==" + version for name, version in contract["runtime"].items()]
).env({"PYTHONPATH": "/execution/code:/execution/inference/code", "PYTHONDONTWRITEBYTECODE": "1",
       "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
if modal.is_local():
    runner.verify(PACKAGE, runner.sha(PACKAGE / "execution.json"))
    image = image.add_local_dir(str(PACKAGE), "/execution")


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=7200,
              retries=0, max_containers=1, min_containers=0, single_use_containers=True,
              volumes={"/artifacts": artifacts, "/cache": cache.read_only()}, include_source=False)
def evaluate(identifier: str, expected_sha: str, attempt: int = 1):
    import importlib.metadata
    import re
    import shutil
    import time

    started = time.monotonic()
    runner.require(re.fullmatch(r"r5_eval_\d{8}T\d{6}Z", identifier), "Invalid evaluation ID")
    execution, manifest, module, cases, rollouts, held = runner.verify(PACKAGE, expected_sha)
    core, scoring = runner.activate(PACKAGE)
    claims.put(identifier + ":binding", expected_sha, skip_if_exists=True)
    runner.require(claims.get(identifier + ":binding") == expected_sha, "Run already bound to another execution")
    core.claim_attempt(claims, identifier, attempt, modal.current_function_call_id(), runner.terminal)
    artifacts.reload()
    cache.reload()
    folder = Path("/artifacts" + runner.PREFIX) / identifier
    binding = {"version": runner.VERSION, "run_id": identifier, "execution_sha256": expected_sha,
               "inference_sha256": execution["inference_sha256"], "evaluation_sha256": execution["evaluation_sha256"],
               "kind": execution["kind"], "variants": execution["variants"],
               "generation_policy": execution["generation_policy"], "guard_policy": execution["guard_policy"]}
    journal = core.Journal(folder / "journal", binding, artifacts.commit)
    plan = list(runner.slots(cases, rollouts, held, execution["variants"], execution["generation_policy"]))
    runner.require(len(plan) == execution["expected_slots"], "Unexpected evaluation plan")

    def status(state, **extra):
        value = {**binding, "state": state, "attempt": attempt, "call_id": modal.current_function_call_id(),
                 "elapsed_seconds": time.monotonic() - started, "training": False,
                 "expected_slots": len(plan),
                 "semantic_quality_established": False, **extra}
        runner.save(folder / "status.json", value)
        artifacts.commit()
        return value

    status("loading")
    try:
        snapshots = {"execution.json": expected_sha,
                     "context-preflight.json": execution["files"]["context-preflight.json"], **{
            name: checksum for name, checksum in execution["files"].items() if name.startswith("provenance/")}}
        for name, checksum in snapshots.items():
            destination = folder / name
            if destination.exists():
                runner.require(runner.sha(destination) == checksum, "Saved provenance/execution differs")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PACKAGE / name, destination)
        remote = Path(manifest["variants"]["r5_response"]["directory"]).parent / "consumption.json"
        evidence = runner.read(PACKAGE / "provenance/summary.json")
        runner.require(runner.sha(remote) == evidence["consumption_sha256"], "Mounted consumption receipt differs")
        runtime = module.Runtime(PACKAGE / "inference", execution["inference_sha256"])
        identities = {v: runtime.identity(v) for v in execution["variants"]}
        actual_versions = {name: importlib.metadata.version(name) for name in execution["runtime"]}
        runner.require(actual_versions == execution["runtime"], "Installed runtime differs")
        proof = {"binding": binding, "attempt": attempt, "call_id": modal.current_function_call_id(),
                 "runtime": actual_versions, "loading": runtime.loading, "identities": identities,
                 "tokenizer_files": {k: v for k, v in manifest["files"].items() if k.startswith("tokenizer/")},
                 "base_bytes_verified": True, "adapter_bytes_verified": True, "kernel_binding_verified": True,
                 "adapter_tensor_verification": "Performed by pinned Runtime.select before each adapter's first operation"}
        proof_path = folder / "runtime" / ("attempt-" + str(attempt) + ".json")
        runner.require(not proof_path.exists(), "Runtime attempt already exists")
        runner.save(proof_path, proof)
        artifacts.commit()
        # Every resumed result must still have its original runtime/loading receipt.
        for slot in plan:
            record = journal.get(slot["key"])
            if record is not None:
                prior = folder / "runtime" / ("attempt-" + str(record["attempt"]) + ".json")
                runner.require(runner.sha(prior) == record["runtime_proof_sha256"]
                               and runner.read(prior)["binding"] == binding
                               and runner.read(prior)["identities"] == identities, "Saved runtime proof differs")
        status("evaluating", expected_slots=len(plan))
        result = runner.evaluate(runtime, core, scoring, plan, journal, attempt, runner.sha(proof_path),
                                 started + execution["limits"]["soft_seconds"])
        runner.report(folder, plan, journal, evidence, identities,
                      loss_preflight=runner.read(PACKAGE / "context-preflight.json").get("language_loss", {}).get("cases", []))
        return status(**result)
    except Exception as error:
        return status("failed", error_type=type(error).__name__, error=str(error)[:2000])
