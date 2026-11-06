"""Separate private R3 actor; deployment requires later explicit authorization.

GHA_R3_DEMO_BUNDLE and GHA_R3_DEMO_SHA must identify a completed sealed bundle.
The manifest selects the app name. No HTTP endpoint, automatic deploy or fallback.
"""
import os
from pathlib import Path
import sys

import modal

LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[1] if LOCAL else Path("/demo")
sys.path.insert(0, str(ROOT / "scripts"))
from understanding_r2_demo import bound_bundle, require, sha, stream_reply

EXPECTED_SHA = os.environ.get("GHA_R3_DEMO_SHA", "")
BUNDLE = Path(os.environ.get("GHA_R3_DEMO_BUNDLE", "")) if LOCAL else Path("/bundle")
if LOCAL:
    require(os.environ.get("GHA_R3_DEMO_BUNDLE"), "Set GHA_R3_DEMO_BUNDLE")
manifest, inference = bound_bundle(BUNDLE, EXPECTED_SHA)
require(manifest["version"] == "r3-comparison-inference-v1", "Completed R3 bundle required")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    *[name + "==" + version for name, version in manifest["runtime"].items()]
).env({"PYTHONPATH": "/bundle/code:/demo/scripts:/demo/modal", "PYTHONDONTWRITEBYTECODE": "1",
       "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
       "GHA_R3_DEMO_BUNDLE": "/bundle", "GHA_R3_DEMO_SHA": EXPECTED_SHA})
if LOCAL:
    image = image.env({"GHA_R3_DEMO_CODE_SHA": sha(ROOT / "scripts/understanding_r2_demo.py"),
                       "GHA_R3_DEMO_ACTOR_SHA": sha(ROOT / "modal/research_understanding_r3_demo.py")})
    image = image.add_local_dir(str(BUNDLE.resolve()), "/bundle")
    image = image.add_local_file(str(ROOT / "scripts/understanding_r2_demo.py"), "/demo/scripts/understanding_r2_demo.py")
    image = image.add_local_file(str(ROOT / "modal/research_understanding_r3_demo.py"), "/demo/modal/research_understanding_r3_demo.py")

app = modal.App(manifest["actor"]["app_name"])
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)


@app.cls(image=image, gpu="H100", cpu=4, memory=49152, timeout=300, startup_timeout=600,
         min_containers=0, max_containers=1, scaledown_window=120, retries=0,
         volumes={"/cache": cache.read_only(), "/artifacts": artifacts.read_only()}, include_source=False)
class UnderstandingR3Demo:
    @modal.enter()
    def setup(self):
        import threading
        require(sha(ROOT / "scripts/understanding_r2_demo.py") == os.environ["GHA_R3_DEMO_CODE_SHA"], "Demo code changed")
        require(sha(ROOT / "modal/research_understanding_r3_demo.py") == os.environ["GHA_R3_DEMO_ACTOR_SHA"], "Actor code changed")
        self.runtime = None
        self.lock = threading.Lock()

    @modal.method(is_generator=True)
    def stream(self, messages: list[dict], variant: str, expected_sha: str, request_id: str):
        require(expected_sha == EXPECTED_SHA and variant in manifest["variants"], "Requested bundle/variant differs")
        inference.validate_messages(messages)
        with self.lock:
            def runtime():
                if self.runtime is None:
                    self.runtime = inference.Runtime(BUNDLE, EXPECTED_SHA)
                return self.runtime
            iterator = stream_reply(runtime, manifest, messages, variant, EXPECTED_SHA, request_id,
                                    modal.current_function_call_id())
            try:
                yield from iterator
            finally:
                iterator.close()
