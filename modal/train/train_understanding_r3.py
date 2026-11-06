"""Bounded private R3 worker; explicit R2-meaning initialization, no auto retry."""
import os
from pathlib import Path
import sys
import modal

PACKAGE = Path(os.environ["GHA_R3_PACKAGE"]).resolve() if modal.is_local() else Path("/r3-package")
sys.path[:0] = [str(PACKAGE / "r3/scripts"), str(PACKAGE / "reused/scripts")]
from run_understanding_r3 import VERSION, RUNTIME, verify_execution

CONTRACT = verify_execution(PACKAGE)
app = modal.App("ghana-understanding-adaptation-r3")
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
claims = modal.Dict.from_name("ghana-understanding-r3-invocations", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    *[name + "==" + version for name, version in RUNTIME.items()]
).env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
       "GHA_R2_SOURCE_ROOT": "/r3-package/frozen", "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONPATH": "/r3-package/r3/modal/train:/r3-package/r3/scripts:/r3-package/reused/scripts:/r3-package/frozen/scripts:/r3-package/frozen/modal/train"})
if modal.is_local():
    image = image.add_local_dir(str(PACKAGE), "/r3-package")


def execute(identifier, execution_sha, mode, resume_run=None):
    import importlib.metadata
    import random
    import shutil
    import time
    import numpy as np
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native_model
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    import run_understanding_r2 as r2
    from run_understanding_r3 import (require, sha, read, save, run_id, mounted_root, verify_pack,
        verify_native_rows, verify_initialization, consumption_receipt, json_rows, INITIALIZATION, SOURCE_FILES, OPT_KEYS)

    started = time.monotonic()
    run_id(identifier)
    require(sha(PACKAGE / "execution.json") == execution_sha, "Execution invocation changed")
    contract = verify_execution(PACKAGE)
    require(mode == contract["mode"] and (not resume_run or mode == "train"), "Wrong execution mode")
    require({n: importlib.metadata.version(n) for n in RUNTIME} == RUNTIME, "Installed runtime differs")
    r2.activate(PACKAGE / "frozen")
    from afrique_instruction_baseline_core import WEIGHT, verify_weight, verify_loading, verify_cache_receipt
    from check_afrique_training_contract import ASSET_HASHES, MODEL, REVISION
    from native_adaptation_run_core import require_kernel_binding
    from response_resume_state import trainable_state_dict, _read_checkpoint
    from understanding_training_core import train_schedule, assert_checkpoint_equivalent
    from understanding_recovery import select_recovery
    from adapter_checkpoint import save_adapter

    manifest, rows, stages = verify_pack(PACKAGE / "inputs")
    stage, opt = stages[0], manifest["optimization"]
    require(claims.put(identifier, {"call_id": modal.current_function_call_id(), "execution_sha256": execution_sha},
                       skip_if_exists=True), "Run already claimed; never resubmit")
    artifacts.reload(); cache.reload()
    root = mounted_root("/artifacts")
    destination = root / identifier
    destination.mkdir(exist_ok=False)
    results, adapter_records, selection = [], [], None

    def record(name, value):
        save(destination / name, value)
        artifacts.commit()
        return value

    def status(state, **extra):
        if (destination / "consumption.json").is_file():
            extra["consumption_sha256"] = sha(destination / "consumption.json")
        return record("status.json", {"state": state, "mode": mode, "run_id": identifier,
            "execution_sha256": execution_sha, "data_sha256": contract["data_sha256"],
            "remote_directory": "/artifacts/" + VERSION + "/" + identifier,
            "physical_directory": str(destination), "elapsed_seconds": time.monotonic() - started,
            "training_completed": False, "production_promoted": False, "initialization": INITIALIZATION, **extra})

    def checkpoint_commit(output, schedule, accumulation, probe=None):
        # Read the sealed full state, not possibly uncommitted exception metrics.
        last = None
        def commit():
            nonlocal last
            reports = [read(p) for p in output.glob("attempt-*/report.json")]
            require(reports, "Checkpoint commit missing progress report")
            report = max(reports, key=lambda r: (r["checkpoint_completed_updates"], r["checkpoint"]))
            checkpoint = Path(report["checkpoint"])
            require(checkpoint.resolve() == checkpoint and checkpoint.parent.parent == output
                    and report["run_sha256"] == execution_sha and report["data_sha256"] == contract["data_sha256"], "Consumption checkpoint outside this run")
            if str(checkpoint) != last:
                proof = r2.checkpoint_proof(checkpoint, execution_sha, contract["data_sha256"])
                state = _read_checkpoint(checkpoint, read(checkpoint / "manifest.json")["identity"])
                require(state["position"]["completed_updates"] == report["checkpoint_completed_updates"], "Checkpoint consumption cursor differs")
                receipt = consumption_receipt(rows, {"schedule": schedule}, accumulation, state["position"],
                    snapshots=stage["admitted_dataset_snapshots"], mode=mode, execution_sha=execution_sha,
                    data_sha=contract["data_sha256"], checkpoint=proof)
                del state
                receipt["run_id"] = identifier
                if probe:
                    receipt["smoke_probe"] = probe
                    save(destination / ("consumption-smoke-" + probe + ".json"), receipt)
                save(destination / "consumption.json", receipt)
                last = str(checkpoint)
            artifacts.commit()
        return commit

    try:
        shutil.copyfile(PACKAGE / "execution.json", destination / "execution.json")
        for name in ("inputs", "frozen", "r3", "reused"):
            shutil.copytree(PACKAGE / name, destination / name)
        record("stages.json", results)
        record("adapters.json", adapter_records)
        record("consumption.json", {"run_id": identifier, **consumption_receipt(rows, stage, opt["grad_accum"],
            {"completed_microbatches": 0, "completed_updates": 0, "microbatches_in_update": 0,
             "exposure": {"row_visits": {}, "input_tokens": 0, "effective_target_tokens": 0}},
            snapshots=stage["admitted_dataset_snapshots"], mode=mode, execution_sha=execution_sha,
            data_sha=contract["data_sha256"])})
        status("loading")
        if mode == "train":
            proof = contract["smoke"]
            smoke_run = root / run_id(proof["run_id"])
            require(smoke_run.resolve() == smoke_run, "Unsafe smoke directory")
            for name, field in (("execution.json", "execution_sha256"), ("smoke.json", "smoke_sha256"), ("status.json", "status_sha256")):
                require(sha(smoke_run / name) == proof[field], "Remote smoke evidence changed")
            from run_understanding_r3 import compare_smoke
            require(compare_smoke(smoke_run, contract) == proof, "Remote smoke does not bind this training")
        if resume_run:
            previous = root / run_id(resume_run)
            require(previous != destination and previous.resolve() == previous, "Unsafe R3 recovery root")
            claim = claims[resume_run]
            outcome = modal.FunctionCall.from_id(claim["call_id"]).get(timeout=0)
            terminal = r2.terminal_pause(claim, outcome, resume_run, execution_sha, contract["data_sha256"])
            require(outcome == read(previous / "status.json"), "Previous terminal status differs")
            selection = select_recovery(previous, runs_root=root, execution_sha256=execution_sha,
                data_sha256=contract["data_sha256"], stages=stages, grad_accum=opt["grad_accum"], claim=claim, terminal_call=terminal)
            _, _, selection = r2.inherit_prefix(previous, destination, stages, selection, contract)
            # Earlier pass snapshots retain their immutable original locations.
            adapter_records = read(previous / "adapters.json")
            for saved in adapter_records:
                path = Path(saved["directory"])
                require(path.resolve() == path and path.is_relative_to(root), "Snapshot outside R3")
                actual = r2.publish_adapter(path, saved["checkpoint"]["directory"], execution_sha, contract["data_sha256"])
                require(actual["files"] == saved["files"] and actual["checkpoint"] == saved["checkpoint"], "Inherited snapshot changed")
            record("adapters.json", adapter_records)
            record("resume.json", {**selection, "previous_run_id": resume_run, "terminal_call": terminal})
        initializer = verify_initialization("/artifacts")
        native_encoding = verify_native_rows(PACKAGE / "inputs", rows)
        snapshot = Path("/cache/hf/hub", "models--" + MODEL.replace("/", "--"), "snapshots", REVISION)
        verify_cache_receipt(read(snapshot / "AFRIQUE_CACHE_READY.json"))
        for name, checksum in ASSET_HASHES.items():
            require(sha(snapshot / name) == checksum, "Cached base asset changed")
        weight_sha = verify_weight(snapshot / "model.safetensors", WEIGHT)
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as handle:
            shapes = {name: list(handle.get_slice(name).get_shape()) for name in handle.keys()}
        kernels = {"chunk": require_kernel_binding(native_model.torch_chunk_gated_delta_rule, chunk_gated_delta_rule),
                   "recurrent": require_kernel_binding(native_model.torch_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule)}
        base, info = Qwen3_5ForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa",
            key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True)
        loading = verify_loading(info, shapes, {k: list(v.shape) for k, v in base.state_dict().items()})
        require(not any(p.is_meta for p in base.parameters()), "Unmaterialized base weights")
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        base.requires_grad_(False)
        model = PeftModel.from_pretrained(base, str(initializer), adapter_name="default", is_trainable=True,
                                          local_files_only=True, autocast_adapter_dtype=True)
        expected = load_file(str(initializer / "adapter_model.safetensors"), device="cpu")
        actual = get_peft_model_state_dict(model, save_embedding_layers=False)
        require(expected.keys() == actual.keys() and all(torch.equal(actual[k].detach().cpu(), v) for k, v in expected.items()),
                "Initial trainable tensors differ from pinned R2 meaning")
        del actual, expected
        model.peft_config["default"].base_model_name_or_path = MODEL
        model.peft_config["default"].revision = REVISION
        model.enable_input_require_grads(); model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        trainables = {n: p for n, p in model.named_parameters() if p.requires_grad}
        require(trainables and all("lora_" in n for n in trainables) and sum(p.numel() for p in trainables.values()) == 43278336
                and all(p.dtype == torch.float32 for p in trainables.values()), "Unexpected trainable parameters")
        record("loading.json", {**loading, "weight_sha256": weight_sha, "kernels": kernels, "runtime": RUNTIME,
            "initialization": INITIALIZATION, "source_files": SOURCE_FILES, "initial_adapter_values_exact": True,
            "native_encoding": native_encoding, "trainable_parameters": 43278336, "frozen_base": True,
            "r2_optimizer_loaded": False, "r3_full_state_resume": bool(selection), "new_vocabulary": False})
        if mode == "smoke":
            witnesses = json_rows(PACKAGE / "inputs/source-witnesses.jsonl")
            primary = [i for i in stage["schedule"] if rows[i]["collection"] in stage["primary_collections"]
                       and sum(m["role"] == "assistant" for m in witnesses[i].get("messages", [])) >= 2]
            replay = [i for i in stage["schedule"] if rows[i]["collection"] not in stage["primary_collections"]]
            language = [i for i in stage["schedule"] if rows[i]["objective"] == "causal_lm"]
            indices = [max(group, key=lambda i: len(rows[i]["input_ids"])) for group in
                       (primary, replay, language or replay, primary)]
            initial = trainable_state_dict(model)
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            options = {"run_sha256": execution_sha, "data_sha256": contract["data_sha256"],
                       "learning_rate": opt["learning_rate"], "warmup_fraction": opt["warmup_fraction"],
                       "grad_accum": 2, "checkpoint_every": 1}
            status("real_weight_backward_and_recovery")
            continuous = train_schedule(model, rows, indices, output=destination / "uninterrupted",
                commit=checkpoint_commit(destination / "uninterrupted", indices, 2, "uninterrupted"), **options)
            expected = trainable_state_dict(model)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in initial["parameters"]:
                        parameter.copy_(initial["parameters"][name])
                for name, buffer in model.named_buffers():
                    buffer.copy_(initial["buffers"][name])
            model.zero_grad(set_to_none=True)
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
            interrupted = train_schedule(model, rows, indices, output=destination / "interrupted", stop_after_updates=1,
                commit=checkpoint_commit(destination / "interrupted", indices, 2, "interrupted"), **options)
            resumed = train_schedule(model, rows, indices, output=destination / "resumed", resume=Path(interrupted["checkpoint"]),
                commit=checkpoint_commit(destination / "resumed", indices, 2, "resumed"), **options)
            actual = trainable_state_dict(model)
            equal = all(torch.equal(expected[k][n], actual[k][n]) for k in expected for n in expected[k])
            require(equal, "R3 real-weight resume differs bitwise")
            recovery = assert_checkpoint_equivalent(continuous["checkpoint"], resumed["checkpoint"])
            record("smoke.json", {"passed": True, "real_weight_backward": True, "initial_adapter_values_exact": True,
                "resume_parameter_and_buffer_bitwise_equal": equal, "full_state_recovery": recovery,
                "native_encoding": native_encoding, "probe_rows": [rows[i]["id"] for i in indices],
                "uninterrupted": continuous, "interrupted": interrupted, "resumed": resumed, "not_a_research_adapter": True})
            return status("completed", smoke_passed=True)

        resume = Path(selection["checkpoint"]) if selection else None
        cursor = selection["completed_updates"] if selection else 0
        boundaries = stage["snapshot_after_updates"] + [stage["optimizer_updates"]]
        for pass_index, boundary in enumerate(boundaries, 1):
            if boundary <= cursor and boundary != stage["optimizer_updates"]:
                continue
            remaining = contract["soft_seconds"] - (time.monotonic() - started)
            status("training", stage="responses", pass_snapshot=pass_index)
            result = train_schedule(model, rows, stage["schedule"], output=destination / "responses",
                run_sha256=execution_sha, data_sha256=contract["data_sha256"], resume=resume,
                stop_after_updates=boundary, max_seconds=max(0, remaining - 90),
                commit=checkpoint_commit(destination / "responses", stage["schedule"], opt["grad_accum"]),
                **{key: opt[key] for key in OPT_KEYS})
            results = [{"stage": "responses", **result}]
            record("stages.json", results)
            if resume:
                restored = Path(result["attempt"]) / ("update-%08d" % cursor)
                record("restore-proof.json", assert_checkpoint_equivalent(resume, restored))
            cursor = result["position"]["completed_updates"]
            if result["status"] != "completed" and not (result["stop_reason"] == "update_limit" and cursor == boundary):
                return status("paused_budget", stages=results, automatic_restart=False)
            complete = result["status"] == "completed"
            if complete:
                r2.check_stage(results[0], stage)
            target = destination / ("responses-adapter" if complete else f"responses-pass{pass_index}-adapter")
            save_adapter(model, target)
            exported = r2.publish_adapter(target, result["checkpoint"], execution_sha, contract["data_sha256"])
            r2.verify_adapter_tensors(model, [exported])
            adapter_records.append({**exported, "completed_updates": cursor, "initialization": INITIALIZATION})
            record("adapters.json", adapter_records)
            if complete:
                return status("completed", stages=results, adapters=adapter_records, training_completed=True)
            resume = Path(result["checkpoint"])
        raise RuntimeError("Finite schedule ended without a completed export")
    except Exception as error:
        status("failed", stages=results, error_type=type(error).__name__, error=str(error))
        raise


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=900, retries=0,
              min_containers=0, max_containers=1, single_use_containers=True,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts}, include_source=False)
def smoke(identifier: str, execution_sha: str, resume_run: str | None = None):
    return execute(identifier, execution_sha, "smoke", resume_run)


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=10800, retries=0,
              min_containers=0, max_containers=1, single_use_containers=True,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts}, include_source=False)
def train(identifier: str, execution_sha: str, resume_run: str | None = None):
    return execute(identifier, execution_sha, "train", resume_run)
