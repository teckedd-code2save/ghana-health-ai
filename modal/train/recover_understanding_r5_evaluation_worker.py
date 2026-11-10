"""One bounded GPU container; local invocation of the unchanged sealed evaluator."""
import os
from pathlib import Path
import sys

import modal

CONTROLLER = Path(os.environ["GHA_R5_RECOVERY_PACKAGE"]) if modal.is_local() else Path("/recovery")
PACKAGE = Path(os.environ["GHA_R5_EVALUATION_PACKAGE"]) if modal.is_local() else Path("/execution")
sys.path.insert(0, str(CONTROLLER / "code"))
_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    import recover_understanding_r5_evaluation as recovery
    api = recovery.api
    worker = api.module_at(PACKAGE / "code/evaluate_understanding_r5.py", "_r5_recovery_sealed_worker", register=True)
finally:
    sys.dont_write_bytecode = _bytecode
app = modal.App("ghana-understanding-evaluation-r5-recovery-private")


def build_image(contract, package, controller, local):
    # Match the frozen worker's build recipe; lazy file mounts must be last.
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        *[name + "==" + version for name, version in contract["runtime"].items()]
    ).env({"PYTHONPATH": "/recovery/code:/execution/code:/execution/inference/code", "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    if local:
        image = image.add_local_dir(str(package), "/execution")
        image = image.add_local_dir(str(controller), "/recovery")
    return image


image = build_image(worker.contract, PACKAGE, CONTROLLER, modal.is_local())


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=7200,
              retries=0, max_containers=1, min_containers=0, single_use_containers=True,
              volumes={"/artifacts": worker.artifacts, "/cache": worker.cache.read_only()}, include_source=False)
def recover(controller_sha: str):
    return recovery.execute(worker, CONTROLLER, controller_sha, PACKAGE, modal.current_function_call_id())
