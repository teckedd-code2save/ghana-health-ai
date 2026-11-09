"""Separate private R5 actor; no deployment or GPU call on import.

GHA_R5_DEMO_BUNDLE and GHA_R5_DEMO_SHA identify a completed sealed bundle.
Raw and guarded methods share the unchanged R3 decoding and streaming policies.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import modal


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


LOCAL = modal.is_local()
EXPECTED_SHA = os.environ.get("GHA_R5_DEMO_SHA", "")
require(re.fullmatch(r"[a-f0-9]{64}", EXPECTED_SHA), "Set GHA_R5_DEMO_SHA")
require(not LOCAL or os.environ.get("GHA_R5_DEMO_BUNDLE"), "Set GHA_R5_DEMO_BUNDLE")
BUNDLE = (Path(os.environ["GHA_R5_DEMO_BUNDLE"]) if LOCAL else Path("/bundle")).resolve()
require(sha(BUNDLE / "manifest.json") == EXPECTED_SHA, "Bundle identity changed")
manifest = json.loads((BUNDLE / "manifest.json").read_bytes())
require(manifest["version"] == "r5-comparison-inference-v1", "Completed R5 bundle required")
require(not any(p.is_symlink() for p in BUNDLE.rglob("*")), "Linked bundle artifact")
for name, checksum in manifest["files"].items():
    if name.startswith("code/"):
        source = BUNDLE / name
        require(source.resolve().is_relative_to(BUNDLE) and sha(source) == checksum, "Bundled code changed")
require(sha(__file__) == manifest["files"]["code/research_understanding_r5_demo.py"], "Actor code changed")
spec = importlib.util.spec_from_file_location("_r5_actor_inference_" + EXPECTED_SHA, BUNDLE / "code/r5_inference.py")
inference = importlib.util.module_from_spec(spec)
exec(compile(Path(spec.origin).read_bytes(), spec.origin, "exec"), inference.__dict__)
manifest = inference.load_bundle(BUNDLE, EXPECTED_SHA)
# Fail closed on an already imported helper from another tree; never patch it.
for name in ("response_loop_guard", "understanding_r2_demo", "guarded_r3_runtime"):
    if name in sys.modules:
        require(Path(getattr(sys.modules[name], "__file__", "")).resolve() == BUNDLE / "code" / (name + ".py"),
                "Streaming helper already imported from another tree")
    else:
        helper = BUNDLE / "code" / (name + ".py")
        helper_spec = importlib.util.spec_from_file_location(name, helper)
        module = importlib.util.module_from_spec(helper_spec)
        sys.modules[name] = module
        # A deployment-time import must not create bytecode in sealed evidence.
        exec(compile(helper.read_bytes(), str(helper), "exec"), module.__dict__)
sys.path.insert(0, str(BUNDLE / "code"))
from understanding_r2_demo import stream_reply

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    *[name + "==" + version for name, version in manifest["runtime"].items()]
).env({"PYTHONPATH": "/bundle/code:/demo/modal", "PYTHONDONTWRITEBYTECODE": "1",
       "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
       "GHA_R5_DEMO_BUNDLE": "/bundle", "GHA_R5_DEMO_SHA": EXPECTED_SHA})
if LOCAL:
    image = image.add_local_dir(str(BUNDLE), "/bundle")
    image = image.add_local_file(str(BUNDLE / "code/research_understanding_r5_demo.py"),
                                 "/demo/modal/research_understanding_r5_demo.py")

app = modal.App(manifest["actor"]["app_name"])
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)


@app.cls(image=image, gpu="H100", cpu=4, memory=49152, timeout=300, startup_timeout=600,
         min_containers=0, max_containers=1, scaledown_window=120, retries=0,
         volumes={"/cache": cache.read_only(), "/artifacts": artifacts.read_only()}, include_source=False)
class UnderstandingR5Demo:
    @modal.enter()
    def setup(self):
        import threading
        inference.load_bundle(BUNDLE, EXPECTED_SHA)
        self.runtime = None
        self.lock = threading.Lock()

    @modal.method(is_generator=True)
    def stream(self, messages: list[dict], variant: str, expected_sha: str, request_id: str):
        yield from self._stream(messages, variant, expected_sha, request_id, False)

    @modal.method(is_generator=True)
    def stream_guarded(self, messages: list[dict], variant: str, expected_sha: str, request_id: str):
        yield from self._stream(messages, variant, expected_sha, request_id, True)

    def _stream(self, messages, variant, expected_sha, request_id, loop_guard):
        require(expected_sha == EXPECTED_SHA and variant in manifest["variants"], "Requested bundle/variant differs")
        inference.validate_messages(messages)
        with self.lock:
            def runtime():
                if self.runtime is None:
                    self.runtime = inference.Runtime(BUNDLE, EXPECTED_SHA)
                if loop_guard:
                    from types import SimpleNamespace
                    from guarded_r3_runtime import reply
                    return SimpleNamespace(identity=self.runtime.identity,
                        reply=lambda *a, **kw: reply(self.runtime, *a, **kw))
                return self.runtime
            iterator = stream_reply(runtime, manifest, messages, variant, EXPECTED_SHA, request_id,
                                    modal.current_function_call_id(), loop_guard=loop_guard)
            try:
                yield from iterator
            finally:
                iterator.close()
