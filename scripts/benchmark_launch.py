#!/usr/bin/env python3
"""Launch the R6 benchmark detached so the local client can exit without stopping it.

Usage: python scripts/benchmark_launch.py <variant> <limit> [<variant> <limit> ...]
Results persist to /artifacts/understanding-adaptation-r6/benchmark-results/<variant>.json
on the train volume; poll with `modal volume ls`.
"""
import importlib.util
import os
import sys

os.environ["MODAL_PROFILE"] = os.environ.get("MODAL_PROFILE", "createdliving1000")

spec = importlib.util.spec_from_file_location("benchmark_r6", "modal/train/benchmark_r6.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

import modal  # noqa: E402

pairs = [(sys.argv[i], int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)]
with modal.enable_output(), module.app.run(detach=True):
    for variant, limit in pairs:
        call = module.benchmark.spawn(variant, limit)
        print(f"submitted {variant} limit={limit} call_id={call.object_id}", flush=True)
print("all submitted; poll volume for results")
