#!/usr/bin/env python3
"""Explicit one-call CPU preflight on the gateway image; never deploy the gateway.

Requires --execute, a sealed bundle pin and a fresh output directory. The selected
ephemeral app contains only a CPU function with no secrets, volumes or network.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

import research_r3_gateway_core as core

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_gateway(bundle, expected_sha):
    os.environ["GHA_R3_DEMO_BUNDLE"] = str(Path(bundle).resolve())
    os.environ["GHA_R3_DEMO_SHA"] = expected_sha
    path = ROOT / "modal/research_r3_gateway.py"
    name = "research_r3_gateway"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    core.require(set(module.preflight_app.registered_functions) == {"preflight"}, "Unexpected preflight app function")
    return module


async def execute(module, out, receipt):
    call = None
    try:
        async with asyncio.timeout(900):
            async with module.preflight_app.run():
                receipt.update(app_id=module.preflight_app.app_id, image_id=module.image.object_id)
                call = await module.preflight.spawn.aio()
                receipt.update(state="submitted", call_id=call.object_id)
                save(out / "submitted.json", receipt)
                value = await call.get.aio(timeout=420)
                core.require(value["state"] == "passed" and value["identity"]["manifest_sha256"] == receipt["manifest_sha256"]
                             and value["call_id"] == call.object_id and value["gpu_called"] is False
                             and value["actor_called"] is False and value["secrets_attached"] is False
                             and value["gateway_deployed"] is False, "Preflight result differs")
                save(out / "result.json", value)
        save(out / "COMPLETE.json", {"result_sha256": core.demo.sha(out / "result.json"),
                                     "manifest_sha256": receipt["manifest_sha256"], "app_stopped": True})
        return value
    except BaseException as exc:
        if call is not None:
            try:
                await asyncio.wait_for(call.cancel.aio(terminate_containers=False), 10)
            except Exception:
                pass
        save(out / "failure.json", {**receipt, "state": "failed", "error_type": type(exc).__name__,
                                    "error": str(exc)[:2000], "gateway_deployed": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    core.load_bundle(args.bundle, args.expected_sha)
    core.require(not args.out.exists(), "Use a fresh immutable preflight directory")
    os.environ["MODAL_PROFILE"] = args.profile
    module = load_gateway(args.bundle, args.expected_sha)
    args.out.mkdir(parents=True)
    receipt = {"state": "prepared", "version": "r3-gateway-cpu-preflight-v1", "manifest_sha256": args.expected_sha,
               "bundle": str(args.bundle.resolve()), "profile": args.profile,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "runner_sha256": core.demo.sha(Path(__file__)),
               "gateway_code_sha256": {name: core.demo.sha(ROOT / name) for name in module.CODE},
               "gpu": None, "cpu": 1, "memory_mib": 2048, "function_timeout_seconds": 180,
               "secrets": [], "volumes": {}, "network_blocked": True, "gateway_deployed": False}
    save(args.out / "prepared.json", receipt)
    import modal
    with modal.enable_output():
        result = asyncio.run(execute(module, args.out, receipt))
    print(json.dumps({"state": result["state"], "out": str(args.out.resolve()),
                      "call_id": result["call_id"], "identity": result["identity"],
                      "accepted": len(result["accepted"]), "rejected": len(result["rejected"]),
                      "gpu_called": False, "gateway_deployed": False}, indent=2))


if __name__ == "__main__":
    main()
