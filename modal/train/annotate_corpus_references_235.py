"""The cached larger teacher, restricted to English-reference analysis."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, cpu_image, image, prepare_weights

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
REVISION = "e156cb4efae43fbee1a1ab073f946a1377e6b969"
app = modal.App("ghana-corpus-reference-annotation-235")
if modal.is_local():
    for name in ("benchmark_stronger_response.py", "corpus_checkpoint.py", "annotate_corpus_references.py"):
        source = str(Path(__file__).with_name(name))
        image = image.add_local_file(source, "/root/" + name)
        cpu_image = cpu_image.add_local_file(source, "/root/" + name)


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=900, retries=0,
              max_containers=1, volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    return prepare_weights(MODEL, REVISION, "reference-qwen235", True)


@app.function(image=image, gpu="H200:2", cpu=8, memory=196608, timeout=900,
              retries=0, max_containers=1, scaledown_window=60,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def annotate(run_id: str, requests: list[dict]):
    from annotate_corpus_references import annotate_model
    return annotate_model(run_id, requests, MODEL, REVISION, 2, "fp8")
