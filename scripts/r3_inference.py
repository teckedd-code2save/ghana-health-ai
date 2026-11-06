"""Pinned R3 inference contract. Imports and validation are CPU/offline only."""
import importlib.util
from pathlib import Path, PurePosixPath
import re

# Resolve the sibling runtime explicitly, not a different bundle in sys.modules.
_spec = importlib.util.spec_from_file_location("_r3_pinned_r2_runtime", Path(__file__).with_name("r2_inference.py"))
base = importlib.util.module_from_spec(_spec)
exec(compile(Path(_spec.origin).read_bytes(), _spec.origin, "exec"), base.__dict__)
read, sha, safe, require, digest = base.read, base.sha, base.safe, base.require, base.digest
validate_messages = base.validate_messages
VERSION = "r3-comparison-inference-v1"
VARIANTS = ("base", "r2_meaning", "r3_response")
R2_BUNDLE_SHA = "6168a3ff4e69756a748f271896a133b67887a6ca8565d08029a7ab184ee1d8f3"
R2_DIRECTORY = "/artifacts/understanding-adaptation-r2/understanding_20261105T192103Z/meaning-adapter"
R2_WEIGHTS = "c117f89326dcb73c79feaa75d8927082300e90278f9d40956ef8ef3e7ead0f3e"
R2_CONFIG = "99e6af0094c9233078660f74351e7c68bc7966a22953e0c281fd68698e968175"
INITIALIZATION = {"adapter_directory": R2_DIRECTORY, "adapter_weights_sha256": R2_WEIGHTS,
                  "new_optimizer": True, "new_scheduler": True}
APP_NAME = "ghana-understanding-r3-demo-private"
CLASS_NAME = "UnderstandingR3Demo"
VOLUME = "ghana-health-understand-train"


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def check_parent(parent, checksum):
    require(checksum == R2_BUNDLE_SHA and parent["version"] == "r2-paired-inference-v1",
            "Use the pinned original R2 bundle")
    require(parent["model"] == base.MODEL and parent["revision"] == base.REVISION
            and parent["base_sha256"] == base.BASE_SHA and parent["decoding"] == base.DECODING
            and parent["runtime"] == base.RUNTIME, "Parent inference protocol changed")
    require(parent["variants"]["base"] is None and parent["variants"]["r2_meaning"] == {
        "directory": R2_DIRECTORY, "files": {"adapter_config.json": R2_CONFIG,
        "adapter_model.safetensors": R2_WEIGHTS}}, "R2 meaning identity changed")


def training_proof(receipt, terminal, fetch):
    """Validate trainer metadata through a hash-reporting, local-only fetch callback."""
    identifier = receipt["run_id"]
    require(re.fullmatch(r"understanding_\d{8}T\d{6}Z", identifier), "Invalid R3 run ID")
    root = "/artifacts/understanding-adaptation-r3/" + identifier
    require(receipt["state"] == "submitted" and receipt["mode"] == "train"
            and receipt["volume"] == VOLUME and receipt["remote_directory"] == root.removeprefix("/artifacts")
            and isinstance(receipt.get("call_id"), str) and receipt["call_id"].startswith("fc-"),
            "Expected submitted R3 training receipt")
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
    require(status["remote_directory"] == root and execution["version"] == data["version"] == "understanding-adaptation-r3"
            and execution["runtime"] == loading["runtime"] == base.RUNTIME
            and loading["weight_sha256"] == base.BASE_SHA, "Wrong R3 foundation/runtime")
    require(all(v["initialization"] == INITIALIZATION for v in (status, execution, data, loading))
            and loading["initial_adapter_values_exact"] is True and loading["frozen_base"] is True
            and loading["r2_optimizer_loaded"] is False and loading["new_vocabulary"] is False,
            "Unverified R2 initialization")
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
            and physical.parts[-2:] == ("understanding-adaptation-r3", identifier), "Invalid physical run directory")

    def remote(path):
        path = PurePosixPath(path)
        require(".." not in path.parts and path.is_relative_to(physical), "Evidence outside final run")
        return root + "/" + path.relative_to(physical).as_posix()

    finals = [e for e in exports if e["stage"] == "responses"]
    require(len(finals) == 1, "Exactly one final response export required")
    export = finals[0]
    require(remote(export["directory"]) == root + "/responses-adapter"
            and export["checkpoint_adapter_tensors_verified"] is True
            and export["completed_updates"] == position["completed_updates"]
            and export["initialization"] == INITIALIZATION, "Unverified final response export")
    require({"adapter_config.json", "adapter_model.safetensors"} <= set(export["files"])
            <= {"adapter_config.json", "adapter_model.safetensors", "README.md"}
            and all(is_sha(v) for v in export["files"].values())
            and export["files"]["adapter_model.safetensors"] != R2_WEIGHTS, "Missing or unchanged R3 adapter")
    config, checksum = fetch(root + "/responses-adapter/adapter_config.json")
    base.adapter_config(config)
    require(checksum == export["files"]["adapter_config.json"], "R3 config readback differs")
    proof = export["checkpoint"]
    require(actual["checkpoint"] == proof["directory"], "Final checkpoint differs")
    checkpoint = remote(proof["directory"])
    cp_manifest, cp_sha = fetch(checkpoint + "/manifest.json")
    committed, _ = fetch(checkpoint + "/COMMITTED.json")
    contract, contract_sha = fetch(str(PurePosixPath(checkpoint).parent.parent) + "/contract.json")
    require(cp_sha == proof["manifest_sha256"] and committed == {"manifest_sha256": cp_sha}
            and cp_manifest["state_sha256"] == proof["state_sha256"] and is_sha(proof["state_sha256"])
            and contract_sha == proof["contract_sha256"] and contract["run_sha256"] == execution_sha
            and contract["data_sha256"] == data_sha
            and cp_manifest["identity"] == {"run_sha256": digest(contract), "data_sha256": data_sha},
            "Checkpoint commitment differs")
    return {"run_id": identifier, "call_id": receipt["call_id"], "execution_sha256": execution_sha,
            "data_sha256": data_sha, "checkpoint": proof, "training_completed": True,
            "completed_updates": position["completed_updates"],
            "variant": {"directory": root + "/responses-adapter", "files": {
                key: export["files"][key] for key in ("adapter_config.json", "adapter_model.safetensors")}}}


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
            "R3 inference protocol changed")
    require(set(manifest["variants"]) == set(VARIANTS) and manifest["variants"]["base"] is None
            and manifest["default_variant"] == "r3_response" and manifest["private"] is True
            and manifest["training"] is False, "Wrong R3 checkpoint set or privacy")
    require({"code/r2_inference.py", "code/r3_inference.py", "r2-manifest.json", "evidence/receipt.json",
             "evidence/terminal.json", "evidence/adapter-readback.json"} <= set(manifest["files"]),
            "Missing required bundle evidence")
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Linked bundle artifact")
    require({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            - {"manifest.json", "SEALED.json"} == set(manifest["files"]), "Unbound or missing bundle file")
    for name, checksum in manifest["files"].items():
        require(sha(safe(folder, name)) == checksum, "Bundle artifact changed: " + name)
    parent = read(folder / "r2-manifest.json")
    parent_sha = sha(folder / "r2-manifest.json")
    check_parent(parent, parent_sha)
    require(manifest["r2_bundle_sha256"] == parent_sha
            and manifest["variants"]["r2_meaning"] == parent["variants"]["r2_meaning"], "Changed parent binding")
    for name, checksum in parent["files"].items():
        if name.startswith("tokenizer/"):
            require(manifest["files"].get(name) == checksum, "Native tokenizer/template changed")
    remote_files = dict(parent["remote_files"])

    def fetch(path):
        name = "evidence/readback/" + path.removeprefix("/artifacts/")
        require(name in manifest["files"], "Unbound R3 training evidence")
        checksum = sha(safe(folder, name))
        remote_files[path] = checksum
        return read(folder / name), checksum

    proof = training_proof(read(folder / "evidence/receipt.json"), read(folder / "evidence/terminal.json"), fetch)
    require(manifest["r3_training"] == proof and manifest["variants"]["r3_response"] == proof["variant"],
            "R3 receipt/variant binding differs")
    readback = read(folder / "evidence/adapter-readback.json")
    require(readback["files"] == proof["variant"]["files"] and readback["directory"] == proof["variant"]["directory"]
            and readback["bytes_read"]["adapter_model.safetensors"] > 0 and readback["verified"] is True,
            "Missing adapter readback proof")
    require(manifest["remote_files"] == remote_files, "Remote evidence binding differs")
    actor = manifest["actor"]
    require(re.fullmatch(r"[a-z][a-z0-9-]{1,62}", actor["app_name"])
            and actor["app_name"] != "ghana-understanding-r2-demo-private"
            and actor["class_name"] == CLASS_NAME, "Use a separate private R3 actor")
    return manifest


class Runtime(base.Runtime):
    def __init__(self, bundle_dir, expected_sha, artifacts_root=Path("/artifacts")):
        super().__init__(bundle_dir, expected_sha, artifacts_root, bundle_loader=load_bundle)
