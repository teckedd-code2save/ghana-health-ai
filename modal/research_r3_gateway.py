"""Separate authenticated R3 SSE bridge; importing does not deploy or call GPU.

Future deployment requires GHA_R3_DEMO_BUNDLE and GHA_R3_DEMO_SHA, exactly as for
UnderstandingR3Demo. The completed sealed bundle supplies the actor target and
all inference identities. Never substitute an R1 execution hash or an R2 bundle.

Provision the named Modal Secret `ghana-research-r3-gateway` separately with
RESEARCH_R3_TOKEN (32+ printable ASCII characters, cryptographically random).
No credential is read locally or baked into the image. Normal app defaults and
the private GPU actor are unchanged. This file deploys only a CPU HTTPS service.

GET /health authenticates configuration identity, NOT upstream readiness.
POST /stream takes Bearer auth and exactly {messages, variant: 'r3_response'}.
It emits data-only start/delta/final objects in the actor's r2-demo-stream-v1
schema, optional SSE comments, or a generic error. Abort the connection to stop.
Before start, no public Modal generator handle exists: cancellation cannot be
confirmed. The gateway temporarily blocks new submissions in that case.
"""
import json
import os
from pathlib import Path
import sys

import modal

LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[1] if LOCAL else Path("/gateway")
sys.path.insert(0, str(ROOT / "scripts"))
import research_r3_gateway_core as core

EXPECTED_SHA = os.environ.get("GHA_R3_DEMO_SHA", "")
BUNDLE = Path(os.environ.get("GHA_R3_DEMO_BUNDLE", "")) if LOCAL else Path("/bundle")
if LOCAL:
    core.require(bool(os.environ.get("GHA_R3_DEMO_BUNDLE")), "Set GHA_R3_DEMO_BUNDLE")
manifest, _ = core.load_bundle(BUNDLE, EXPECTED_SHA)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.136.1", "starlette==1.2.1", "anyio==4.12.1", "jinja2==3.1.6",
    "transformers==" + manifest["runtime"]["transformers"]
).env({"PYTHONPATH": "/gateway/modal:/gateway/scripts:/bundle/code", "PYTHONDONTWRITEBYTECODE": "1",
       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "DO_NOT_TRACK": "1", "GHA_R3_DEMO_BUNDLE": "/bundle", "GHA_R3_DEMO_SHA": EXPECTED_SHA})
CODE = ("modal/research_r3_gateway.py", "scripts/research_r3_gateway_core.py", "scripts/understanding_r2_demo.py")
if LOCAL:
    image = image.env({"GHA_R3_GATEWAY_CODE_SHA": json.dumps({name: core.demo.sha(ROOT / name) for name in CODE}, sort_keys=True)})
    image = image.add_local_dir(str(BUNDLE.resolve()), "/bundle")
    for name in CODE:
        image = image.add_local_file(str(ROOT / name), "/gateway/" + name)

app = modal.App("ghana-research-r3-gateway")
preflight_app = modal.App("ghana-research-r3-gateway-cpu-preflight")


def remote(messages, variant, expected_sha, request_id):
    target = manifest["actor"]
    actor = modal.Cls.from_name(target["app_name"], target["class_name"])()
    return actor.stream.remote_gen.aio(messages, variant, expected_sha, request_id)


async def cancel(call_id):
    await modal.FunctionCall.from_id(call_id).cancel.aio(terminate_containers=False)


@app.function(image=image, cpu=1, memory=2048, timeout=960, min_containers=0, max_containers=1,
              scaledown_window=960, include_source=False,
              secrets=[modal.Secret.from_name("ghana-research-r3-gateway", required_keys=["RESEARCH_R3_TOKEN"])])
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def gateway():
    expected = core.parse_json(os.environ["GHA_R3_GATEWAY_CODE_SHA"])
    core.require(expected == {name: core.demo.sha(ROOT / name) for name in CODE}, "Gateway code changed")
    contract = core.load_contract(BUNDLE, EXPECTED_SHA)
    return core.create_api(os.environ["RESEARCH_R3_TOKEN"], contract, remote, cancel)


@preflight_app.function(image=image, cpu=1, memory=2048, timeout=180, startup_timeout=180,
                        min_containers=0, max_containers=1, retries=0, single_use_containers=True,
                        secrets=[], volumes={}, block_network=True, restrict_modal_access=True,
                        include_source=False)
def preflight():
    expected = core.parse_json(os.environ["GHA_R3_GATEWAY_CODE_SHA"])
    core.require(expected == {name: core.demo.sha(ROOT / name) for name in CODE}, "Gateway code changed")
    core.require(not Path("/dev/nvidia0").exists(), "CPU preflight must not have a GPU")
    return {**core.cpu_preflight(BUNDLE, EXPECTED_SHA), "gateway_code_sha256": expected,
            "call_id": modal.current_function_call_id(), "secrets_attached": False,
            "network_blocked": True, "gateway_deployed": False}
