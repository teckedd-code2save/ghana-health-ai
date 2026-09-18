"""Newer, single-GPU foundation comparison. Separate app, receipts and artifacts."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, cpu_image, image, prepare_weights, compare_model

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
app = modal.App("ghana-current-response-comparison")

source = str(Path(__file__).with_name("benchmark_stronger_response.py"))
if modal.is_local():
    cpu_image = cpu_image.add_local_file(source, "/root/benchmark_stronger_response.py")
    image = image.add_local_file(source, "/root/benchmark_stronger_response.py")


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=1200, retries=0, max_containers=1,
              volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    return prepare_weights(MODEL, REVISION, "qwen38-27b", False)


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=1800, retries=0, max_containers=1,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def compare(run_id: str, cases: list[dict]):
    return compare_model(run_id, cases, MODEL, REVISION, "qwen38", 1, None, reasoning_effort="medium")
