"""Pinned R6 inference contract. Imports and validation are CPU/offline only."""
import importlib.util
from pathlib import Path, PurePosixPath
import re

# Resolve the sibling runtime explicitly, not a different bundle in sys.modules.
_spec = importlib.util.spec_from_file_location("_r6_pinned_r2_runtime", Path(__file__).with_name("r2_inference.py"))
base = importlib.util.module_from_spec(_spec)
exec(compile(Path(_spec.origin).read_bytes(), _spec.origin, "exec"), base.__dict__)
read, sha, safe, require, digest = base.read, base.sha, base.safe, base.require, base.digest
validate_messages = base.validate_messages
VERSION = "r6-comparison-inference-v1"
VARIANTS = ("base", "r5_response", "r6_response")
R5_BUNDLE_SHA = "6e232b799c622fa55885baa1f76d5d86b1388974144ae16ddf5f5b9ef02076a6"
R5_DIRECTORY = "/artifacts/understanding-adaptation-r5/understanding_20261110T103734Z/responses-adapter"
R5_WEIGHTS = "2085ee881927ee559d4bd978ae2372e12dea963d588fd136b2130bc551835d25"
R5_CONFIG = "e66b69cf23b576fb5dd9e5b6a8f4156a14983ec7599fc60a0188f553eab4ab0f"
R6_DIRECTORY = "/artifacts/understanding-adaptation-r6/understanding_20261113T152209Z/responses-adapter"
R6_WEIGHTS = "6e56d8c7c04f456f94579d4bd82064521c97d0b9e23dbbef1e9783280568861c"
R6_CONFIG = "577b80656dbcff84886c4e402022d9507abbef34b0e0013bceab029ccf4c9403"
INITIALIZATION = {"adapter_directory": R5_DIRECTORY, "adapter_weights_sha256": R5_WEIGHTS,
                  "new_optimizer": True, "new_scheduler": True}
APP_NAME = "ghana-understanding-r6-demo-private"
CLASS_NAME = "UnderstandingR6Demo"
VOLUME = "ghana-health-understand-train"
# Exact, unmodified helpers from the dirty R2 worktree, sealed into each bundle.
HELPER_HASHES = {
    "understanding_r2_demo.py": "e11649ff095b6315a06b02ac72661e4e4006aefb73c9eceb8e851539ce127781",
    "guarded_r3_runtime.py": "78b67c116cf6edd9b43636c52519ee294847fe41db0077feb8101aef9e3bfb92",
    "response_loop_guard.py": "2b0830e33dd3a9bea182d2dbbd4d9d8c091f14eaea4b23e95a6664391ac3b474",
}


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def check_parent(parent, checksum):
    require(checksum == R5_BUNDLE_SHA and parent["version"] == "r5-comparison-inference-v1",
            "Use the pinned original R5 bundle")
    require(parent["model"] == base.MODEL and parent["revision"] == base.REVISION
            and parent["base_sha256"] == base.BASE_SHA and parent["decoding"] == base.DECODING
            and parent["runtime"] == base.RUNTIME, "Parent inference protocol changed")
    require(parent["variants"]["base"] is None and parent["variants"]["r5_response"] == {
        "directory": R5_DIRECTORY, "files": {"adapter_config.json": R5_CONFIG,
        "adapter_model.safetensors": R5_WEIGHTS}}, "R5 response identity changed")


def training_proof(receipt, terminal, fetch):
    """Validate trainer metadata through a hash-reporting, local-only fetch callback."""
    identifier = receipt["run_id"]
    require(identifier == "understanding_20261113T152209Z", "Invalid R6 run ID")
    root = "/artifacts/understanding-adaptation-r6/" + identifier
    require(receipt["state"] == "submitted" and receipt["mode"] == "train"
            and receipt["volume"] == VOLUME and receipt["remote_directory"] == root.removeprefix("/artifacts")
            and isinstance(receipt.get("call_id"), str) and receipt["call_id"].startswith("fc-"),
            "Expected submitted R6 training receipt")
    require(is_sha(receipt["execution_sha256"]) and is_sha(receipt["data_sha256"]), "Invalid training hashes")
    status, _ = fetch(root + "/status.json")
    execution, execution_sha = fetch(root + "/execution.json")
    data, data_sha = fetch(root + "/inputs/manifest.json")
    loading, _ = fetch(root + "/loading.json")
    exports, _ = fetch(root + "/adapters.json")
    require(status == terminal and status.get("state") == "completed" and status.get("training_completed") is True
            and status["mode"] == execution["mode"] == "train" and status["run_id"] == identifier,
            "Matching completed terminal training receipt required")
    require(execution_sha == status["execution_sha256"] == receipt["execution_sha256"]
            and data_sha == status["data_sha256"] == execution["data_sha256"] == receipt["data_sha256"],
            "Execution/data receipt identity differs")
    require(status["remote_directory"] == root and execution["version"] == data["version"] == "understanding-adaptation-r6"
            and execution["runtime"] == loading["runtime"] == base.RUNTIME
            and loading["weight_sha256"] == base.BASE_SHA, "Wrong R6 foundation/runtime")
    require(all(v["initialization"] == INITIALIZATION for v in (status, execution, data, loading))
            and loading["initial_adapter_values_exact"] is True and loading["frozen_base"] is True
            and loading["r4_optimizer_loaded"] is False and loading["r5_full_state_resume"] is False
            and loading["new_vocabulary"] is False,
            "Unverified R5 initialization")
    require(exports == status["adapters"] and len(status["stages"]) == len(execution["expected_stages"]) == 1,
            "Incomplete stage/export evidence")
    actual, planned = status["stages"][0], execution["expected_stages"][0]
    position = actual["position"]
    require(actual["stage"] == planned["name"] == "responses" and actual["status"] == "completed"
            and actual["completion_verified"] is True and planned["optimizer_updates"] > 0
            and position["completed_updates"] == planned["optimizer_updates"]
            and position["completed_microbatches"] == planned["scheduled_sequences"]
            and position["microbatches_in_update"] == 0
            and position["exposure"]["effective_target_tokens"] == planned["effective_target_tokens"]
            and position["exposure"]["input_tokens"] == planned["sequence_tokens"], "Incomplete final training exposure")
    physical = PurePosixPath(status["physical_directory"])
    require(physical.is_absolute() and ".." not in physical.parts
            and physical.parts[-2:] == ("understanding-adaptation-r6", identifier), "Invalid physical run directory")

    def remote(path):
        path = PurePosixPath(path)
        require(".." not in path.parts and path.is_relative_to(physical), "Evidence outside final run")
        return root + "/" + path.relative_to(physical).as_posix()

    def export_proof(stage, updates):
        candidates = [e for e in exports if e["stage"] == stage]
        require(len(candidates) == 1, "Exactly one " + stage + " export required")
        export = candidates[0]
        directory = root + "/" + stage + "-adapter"
        require(remote(export["directory"]) == directory
                and export["checkpoint_adapter_tensors_verified"] is True
                and export["completed_updates"] == updates
                and export["initialization"] == INITIALIZATION, "Unverified response export")
        require({"adapter_config.json", "adapter_model.safetensors"} <= set(export["files"])
                <= {"adapter_config.json", "adapter_model.safetensors", "README.md"}
                and all(is_sha(v) for v in export["files"].values())
                and export["files"]["adapter_model.safetensors"] != R5_WEIGHTS, "Missing or unchanged R6 adapter")
        config, checksum = fetch(directory + "/adapter_config.json")
        base.adapter_config(config)
        require(checksum == export["files"]["adapter_config.json"], "R6 config readback differs")
        proof = export["checkpoint"]
        checkpoint = remote(proof["directory"])
        require(PurePosixPath(checkpoint).name == "update-%08d" % updates, "Checkpoint update differs")
        cp_manifest, cp_sha = fetch(checkpoint + "/manifest.json")
        committed, _ = fetch(checkpoint + "/COMMITTED.json")
        contract, contract_sha = fetch(str(PurePosixPath(checkpoint).parent.parent) + "/contract.json")
        require(cp_sha == proof["manifest_sha256"] and committed == {"manifest_sha256": cp_sha}
                and cp_manifest["state_sha256"] == proof["state_sha256"] and is_sha(proof["state_sha256"])
                and contract_sha == proof["contract_sha256"] and contract["run_sha256"] == execution_sha
                and contract["data_sha256"] == data_sha
                and cp_manifest["identity"] == {"run_sha256": digest(contract), "data_sha256": data_sha},
                "Checkpoint commitment differs")
        return {"checkpoint": proof, "completed_updates": updates,
                "variant": {"directory": directory, "files": {
                    key: export["files"][key] for key in ("adapter_config.json", "adapter_model.safetensors")}}}

    final = export_proof("responses", position["completed_updates"])
    require(final["variant"] == {"directory": R6_DIRECTORY, "files": {
        "adapter_config.json": R6_CONFIG, "adapter_model.safetensors": R6_WEIGHTS}},
        "Pinned R6 adapter differs")
    require(actual["checkpoint"] == final["checkpoint"]["directory"], "Final checkpoint differs")
    consumption, consumption_sha = fetch(root + "/consumption.json")
    require(consumption_sha == status["consumption_sha256"]
            and consumption["version"] == "r5-consumption-v1" and consumption["run_id"] == identifier
            and consumption["mode"] == "train" and consumption["training_complete"] is True
            and consumption["schedule_complete"] is True and consumption["execution_sha256"] == execution_sha
            and consumption["data_sha256"] == data_sha and consumption["checkpoint"] == final["checkpoint"]
            and consumption["completed_updates"] == final["completed_updates"]
            and consumption["consumed"]["row_exposures"] == planned["scheduled_sequences"]
            and consumption["consumed"]["effective_target_tokens"] == planned["effective_target_tokens"]
            and consumption["consumed"]["input_tokens"] == planned["sequence_tokens"], "Consumption proof differs")
    result = {"run_id": identifier, "call_id": receipt["call_id"], "execution_sha256": execution_sha,
              "data_sha256": data_sha, "training_completed": True, "consumption_sha256": consumption_sha, **final}
    return result


def load_bundle(folder, expected_sha=None):
    folder = Path(folder)
    if expected_sha is not None:
        require(sha(folder / "manifest.json") == expected_sha, "Bundle identity changed")
    if (folder / "SEALED.json").exists():
        require(read(folder / "SEALED.json") == {"manifest_sha256": sha(folder / "manifest.json")}, "Bundle seal differs")
    manifest = read(folder / "manifest.json")
    require(manifest["version"] == VERSION and manifest["model"] == base.MODEL
            and manifest["revision"] == base.REVISION and manifest["base_sha256"] == base.BASE_SHA
            and manifest["runtime"] == base.RUNTIME and manifest["decoding"] == base.DECODING,
            "R6 inference protocol changed")
    require(set(manifest["variants"]) == set(VARIANTS)
            and manifest["variants"]["base"] is None
            and manifest["default_variant"] == "r6_response" and manifest["private"] is True
            and manifest["training"] is False, "Wrong R6 checkpoint set or privacy")
    require({"code/r2_inference.py", "code/r6_inference.py", "r5-manifest.json", "evidence/receipt.json",
             "evidence/terminal.json", "evidence/adapter-readback.json", "code/research_understanding_r6_demo.py"}
            | {"code/" + name for name in HELPER_HASHES} <= set(manifest["files"]),
            "Missing required bundle evidence")
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Linked bundle artifact")
    require({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            - {"manifest.json", "SEALED.json"} == set(manifest["files"]), "Unbound or missing bundle file")
    for name, checksum in manifest["files"].items():
        require(sha(safe(folder, name)) == checksum, "Bundle artifact changed: " + name)
    parent = read(folder / "r5-manifest.json")
    parent_sha = sha(folder / "r5-manifest.json")
    check_parent(parent, parent_sha)
    require(manifest["r5_bundle_sha256"] == parent_sha
            and manifest["variants"]["r5_response"] == parent["variants"]["r5_response"], "Changed parent binding")
    for name, checksum in parent["files"].items():
        if name.startswith(("tokenizer/", "code/")):
            require(manifest["files"].get(name) == checksum, "Inherited tokenizer/runtime changed")
    require(all(manifest["files"]["code/" + name] == checksum for name, checksum in HELPER_HASHES.items()),
            "Pinned streaming/guard helper changed")
    remote_files = {}

    def fetch(path):
        name = "evidence/readback/" + path.removeprefix("/artifacts/")
        require(name in manifest["files"], "Unbound R6 training evidence")
        checksum = sha(safe(folder, name))
        remote_files[path] = checksum
        return read(folder / name), checksum

    proof = training_proof(read(folder / "evidence/receipt.json"), read(folder / "evidence/terminal.json"), fetch)
    require(manifest["r6_training"] == proof and manifest["variants"]["r6_response"] == proof["variant"],
            "R6 receipt/variant binding differs")
    expected_adapters = {"r5_response": parent["variants"]["r5_response"], "r6_response": proof["variant"]}
    readbacks = read(folder / "evidence/adapter-readback.json")
    require(set(readbacks) == set(expected_adapters), "Missing adapter readback proof")
    for name, adapter in expected_adapters.items():
        readback = readbacks[name]
        require(manifest["variants"][name] == adapter and readback["files"] == adapter["files"]
                and readback["directory"] == adapter["directory"]
                and all(type(readback["bytes_read"][key]) is int and readback["bytes_read"][key] > 0 for key in adapter["files"])
                and readback["verified"] is True, "Missing adapter readback proof")
    require(manifest["remote_files"] == remote_files, "Remote evidence binding differs")
    actor = manifest["actor"]
    require(re.fullmatch(r"[a-z][a-z0-9-]{1,62}", actor["app_name"])
            and actor["app_name"] not in {"ghana-understanding-r2-demo-private", "ghana-understanding-r3-demo-private",
                                          "ghana-understanding-r4-demo-private", "ghana-understanding-r5-demo-private"}
            and actor["class_name"] == CLASS_NAME, "Use a separate private R6 actor")
    return manifest


def bound_bundle(folder, expected_sha):
    """Verify all code before importing the bundle's runtime, without bytecode writes."""
    folder = Path(folder).resolve()
    require(is_sha(expected_sha) and sha(folder / "manifest.json") == expected_sha, "Bundle identity changed")
    manifest = read(folder / "manifest.json")
    require({"code/r6_inference.py", "code/r2_inference.py"} <= set(manifest["files"]), "Missing bundled runtime")
    for name, checksum in manifest["files"].items():
        if name.startswith("code/"):
            require(sha(safe(folder, name)) == checksum, "Bundled inference code changed")
    source = safe(folder, "code/r6_inference.py")
    spec = importlib.util.spec_from_file_location("_r6_bundle_" + expected_sha, source)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return module.load_bundle(folder, expected_sha), module


class Runtime(base.Runtime):
    def __init__(self, bundle_dir, expected_sha, artifacts_root=Path("/artifacts")):
        super().__init__(bundle_dir, expected_sha, artifacts_root, bundle_loader=load_bundle)
