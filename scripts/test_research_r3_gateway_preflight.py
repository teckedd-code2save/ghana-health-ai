"""Offline tests for the isolated CPU preflight; no Modal calls or secrets."""
import ast
from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import run_research_r3_gateway_preflight as runner


class PackagingTests(unittest.TestCase):
    def test_cpu_app_uses_exact_gateway_image_without_secret_volume_gpu_or_network(self):
        tree = ast.parse((runner.ROOT / "modal/research_r3_gateway.py").read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        options = {k.arg: k.value for k in functions["preflight"].decorator_list[0].keywords}
        gateway = {k.arg: k.value for k in functions["gateway"].decorator_list[0].keywords}
        self.assertEqual(ast.unparse(options["image"]), ast.unparse(gateway["image"]))
        self.assertEqual(ast.unparse(functions["preflight"].decorator_list[0].func), "preflight_app.function")
        self.assertEqual(ast.unparse(functions["gateway"].decorator_list[0].func), "app.function")
        self.assertNotIn("gpu", options)
        for key, value in {"cpu": 1, "memory": 2048, "timeout": 180, "startup_timeout": 180,
                           "min_containers": 0, "max_containers": 1, "retries": 0,
                           "single_use_containers": True, "secrets": [], "volumes": {},
                           "block_network": True, "restrict_modal_access": True, "include_source": False}.items():
            self.assertEqual(ast.literal_eval(options[key]), value)
        self.assertNotIn("RESEARCH_R3_TOKEN", ast.unparse(functions["preflight"]))
        paths = [ast.literal_eval(value) for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "env"
                 and isinstance(node.args[0], ast.Dict)
                 for key, value in zip(node.args[0].keys, node.args[0].values)
                 if isinstance(key, ast.Constant) and key.value == "PYTHONPATH"]
        self.assertEqual(len(paths), 1)
        self.assertIn("/gateway/modal", paths[0].split(":"))
        packages = [arg.value for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pip_install"
                    for arg in node.args if isinstance(arg, ast.Constant)]
        self.assertIn("jinja2==3.1.6", packages)

    def test_runner_selects_only_transient_cpu_app(self):
        tree = ast.parse(Path(runner.__file__).read_text())
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertIn("module.preflight_app.run", calls)
        self.assertIn("module.preflight.spawn.aio", calls)
        self.assertNotIn("module.app.run", calls)
        self.assertFalse(any("deploy" in name or "gateway.remote" in name for name in calls))


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def module(self, result):
        self.stopped = False
        @asynccontextmanager
        async def run():
            try:
                yield
            finally:
                self.stopped = True
        call = SimpleNamespace(object_id="fc-cputest", get=SimpleNamespace(aio=AsyncMock(return_value=result)),
                               cancel=SimpleNamespace(aio=AsyncMock()))
        module = SimpleNamespace(preflight_app=SimpleNamespace(run=run, app_id="ap-cputest"),
                                 image=SimpleNamespace(object_id="im-shared"),
                                 preflight=SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(return_value=call))))
        return module, call

    async def test_success_records_exact_result_and_stops_without_deployment(self):
        value = {"state": "passed", "identity": {"manifest_sha256": "a" * 64}, "call_id": "fc-cputest",
                 "gpu_called": False, "actor_called": False, "secrets_attached": False, "gateway_deployed": False}
        module, call = self.module(value)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            result = await runner.execute(module, out, {"manifest_sha256": "a" * 64})
            self.assertEqual(result, value)
            self.assertEqual(json.loads((out / "result.json").read_text()), value)
            commitment = json.loads((out / "COMPLETE.json").read_text())
            self.assertEqual(commitment["result_sha256"], runner.core.demo.sha(out / "result.json"))
            self.assertTrue(commitment["app_stopped"])
        self.assertTrue(self.stopped)
        module.preflight.spawn.aio.assert_awaited_once_with()
        call.get.aio.assert_awaited_once_with(timeout=420)
        call.cancel.aio.assert_not_awaited()

    async def test_failed_call_is_cancelled_and_recorded_not_retried(self):
        module, call = self.module(None)
        call.get.aio.side_effect = TimeoutError("synthetic timeout")
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with self.assertRaises(TimeoutError):
                await runner.execute(module, out, {"manifest_sha256": "a" * 64})
            self.assertEqual(json.loads((out / "failure.json").read_text())["state"], "failed")
            self.assertFalse((out / "COMPLETE.json").exists())
        self.assertTrue(self.stopped)
        module.preflight.spawn.aio.assert_awaited_once_with()
        call.cancel.aio.assert_awaited_once_with(terminate_containers=False)


if __name__ == "__main__":
    unittest.main()
