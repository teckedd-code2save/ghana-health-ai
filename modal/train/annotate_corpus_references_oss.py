"""Cached open-weight reasoning teacher on Modal. No OpenAI API calls."""
from pathlib import Path
import modal
from benchmark_stronger_response import artifacts, cache, cpu_image, image, prepare_weights

MODEL = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
app = modal.App("ghana-corpus-reference-annotation-oss")
if modal.is_local():
    for name in ("benchmark_stronger_response.py", "corpus_checkpoint.py", "annotate_corpus_references.py", "oss_response_core.py"):
        source = str(Path(__file__).with_name(name))
        image = image.add_local_file(source, "/root/" + name)
        cpu_image = cpu_image.add_local_file(source, "/root/" + name)


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=900, retries=0,
              max_containers=1, volumes={"/cache": cache, "/artifacts": artifacts})
def prepare():
    return prepare_weights(MODEL, REVISION, "reference-oss120", False)


@app.function(image=image, gpu="H100", cpu=8, memory=98304, timeout=900,
              retries=0, max_containers=1, scaledown_window=60,
              volumes={"/cache": cache.read_only(), "/artifacts": artifacts})
def annotate(run_id: str, requests: list[dict]):
    from annotate_corpus_references import annotate_model
    return annotate_model(run_id, requests, MODEL, REVISION, 1, "mxfp4", True)
