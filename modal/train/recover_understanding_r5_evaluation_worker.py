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
image = worker.image.env({"PYTHONPATH": "/recovery/code:/execution/code:/execution/inference/code"})
if modal.is_local():
    image = image.add_local_dir(str(CONTROLLER), "/recovery")


@app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=7200,
              retries=0, max_containers=1, min_containers=0, single_use_containers=True,
              volumes={"/artifacts": worker.artifacts, "/cache": worker.cache.read_only()}, include_source=False)
def recover(controller_sha: str):
    return recovery.execute(worker, CONTROLLER, controller_sha, PACKAGE, modal.current_function_call_id())
