"""Offline R3 UI, routing and shared-history regression tests; no server or GPU."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import r2_comparison as comparison
import r2_inference as inference
import r3_inference as r3
import understanding_r2_demo as demo
import understanding_r2_workbench as ui
from test_understanding_r2_demo import manifest as r2_manifest, SHA
from test_prepare_understanding_r3_inference import fixture
import prepare_understanding_r3_inference as builder


def manifest():
    value = r2_manifest()
    value.update(version=r3.VERSION, default_variant="r3_response",
                 actor={"app_name": r3.APP_NAME, "class_name": r3.CLASS_NAME})
    value["variants"] = {"base": None, "r2_meaning": value["variants"]["r2_meaning"],
                         "r3_response": {"directory": "/artifacts/understanding-adaptation-r3/understanding_20261106T120000Z/responses-adapter",
                                         "files": {"adapter_config.json": "d" * 64, "adapter_model.safetensors": "e" * 64}}}
    return value


class LayoutTests(unittest.TestCase):
    def test_r3_only_appears_for_r3_bundle_and_exact_hash_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            remote = Mock(side_effect=AssertionError("Unexpected remote call"))
            with patch.object(demo, "bound_bundle", return_value=(manifest(), inference)):
                app = ui.build_app(folder, SHA, Path(folder) / "r3.sqlite3", remote=remote)
            remote.assert_not_called()
            self.assertFalse(app.is_running)
            config = app.get_config_file()
            components = config["components"]
            choice = next(c["props"] for c in components if c.get("props", {}).get("label") == "Candidate checkpoint")
            self.assertEqual(choice["value"], "r3_response")
            self.assertEqual({v for _, v in choice["choices"]}, {"r2_meaning", "r3_response"})
            chat = next(c["props"] for c in components if c.get("props", {}).get("label") == "Model")
            self.assertEqual(chat["value"], "r3_response")
            self.assertEqual({v for _, v in chat["choices"]}, set(r3.VARIANTS))
            visible = [c["props"].get("value", "") for c in components if c["type"] == "markdown"]
            self.assertTrue(any("r3_response" in text and "e" * 64 in text for text in visible))
            self.assertTrue(any("Adapter disabled" in text and inference.BASE_SHA in text for text in visible))
            old = r2_manifest()
            self.assertEqual(demo.default_variant(old), "r2_meaning")
            self.assertNotIn("r3_response", demo.available_labels(old))

    def test_r3_default_store_is_separate_and_foreign_store_is_not_modified(self):
        current = manifest()
        self.assertEqual(demo.bundle_store(r2_manifest(), SHA), demo.STORE)
        self.assertNotEqual(demo.bundle_store(current, SHA), demo.STORE)
        self.assertNotEqual(demo.bundle_store(current, SHA), demo.bundle_store(current, "b" * 64))
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / "existing-r2.sqlite3"
            demo.save_turn(store, {"id": "old", "session_id": "r2", "created": 1,
                                  "status": "complete", "identity": {"manifest_sha256": "b" * 64}})
            checksum = demo.sha(store)
            with self.assertRaisesRegex(ValueError, "another bundle"):
                demo.bundle_store(current, SHA, store)
            self.assertEqual(demo.sha(store), checksum)

    def test_runtime_identity_and_base_disabling_use_selected_bundle(self):
        runtime = r3.Runtime.__new__(r3.Runtime)
        runtime.manifest, runtime.manifest_sha = manifest(), SHA
        for variant in r3.VARIANTS:
            self.assertEqual(runtime.identity(variant), demo.identity(manifest(), SHA, variant))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            runtime.identity("r2_response")
        disabled = Mock()
        runtime.loaded, runtime.model = {"r3_response"}, SimpleNamespace(disable_adapter=disabled)
        runtime.select("base")
        disabled.assert_called_once_with()

    def test_actor_is_separate_private_readonly_and_no_launch_path(self):
        source = (demo.ROOT / "modal/research_understanding_r3_demo.py").read_text()
        tree = ast.parse(source)
        actor = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertEqual(actor.name, "UnderstandingR3Demo")
        options = {k.arg: k.value for k in actor.decorator_list[0].keywords}
        for key, value in {"min_containers": 0, "max_containers": 1, "retries": 0,
                           "include_source": False, "gpu": "H100", "timeout": 300}.items():
            self.assertEqual(ast.literal_eval(options[key]), value)
        self.assertEqual(source.count(".read_only()"), 2)
        for forbidden in (".deploy(", ".spawn(", ".remote(", "fastapi_endpoint"):
            self.assertNotIn(forbidden, source)
        self.assertIn('modal.App(manifest["actor"]["app_name"])', source)

    def test_successful_bound_import_uses_sibling_not_global_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, parent_sha, readback, _ = fixture(root)
            bundle = root / "bundle"
            with patch.object(r3, "R2_BUNDLE_SHA", parent_sha):
                builder.prepare(parent, parent_sha, root / "receipt.json", root / "terminal.json", readback, bundle)
            # The synthetic parent pin exists only in this temporary fixture copy.
            source = bundle / "code/r3_inference.py"
            source.write_text(source.read_text().replace(r3.R2_BUNDLE_SHA, parent_sha))
            value = inference.read(bundle / "manifest.json")
            value["files"]["code/r3_inference.py"] = inference.sha(source)
            inference.save(bundle / "manifest.json", value)
            checksum = inference.sha(bundle / "manifest.json")
            inference.save(bundle / "SEALED.json", {"manifest_sha256": checksum})
            with patch.object(sys, "path", list(sys.path)), patch.object(sys, "dont_write_bytecode", False), \
                    patch.dict(sys.modules, {"r2_inference": object()}):
                loaded, module = demo.bound_bundle(bundle, checksum)
            self.assertEqual(loaded["version"], r3.VERSION)
            self.assertFalse(list(bundle.rglob("__pycache__")))
            self.assertEqual(Path(module.base.__file__).resolve(), (bundle / "code/r2_inference.py").resolve())
            runtime = module.Runtime.__new__(module.Runtime)
            runtime.manifest, runtime.manifest_sha = loaded, checksum
            self.assertEqual(runtime.identity("r3_response"), demo.identity(loaded, checksum, "r3_response"))


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_actor_routing_is_per_instance_and_legacy_call_is_unchanged(self):
        calls = []

        def from_name(app, cls):
            async def events(*args):
                calls.append((app, cls, args))
                yield {"fixture": app}
            return lambda: SimpleNamespace(stream=SimpleNamespace(remote_gen=SimpleNamespace(aio=events)))

        legacy, current = r2_manifest(), manifest()
        left = demo.routed_events(legacy)
        right = demo.routed_events(current, app_name="custom-r3-private")
        fake_modal = SimpleNamespace(Cls=SimpleNamespace(from_name=from_name))
        with patch.dict(sys.modules, {"modal": fake_modal}):
            for remote in (right, left, demo.modal_events, right):
                _ = [event async for event in remote([], "base", SHA, "request")]
        self.assertEqual([c[:2] for c in calls], [
            ("custom-r3-private", "UnderstandingR3Demo"), (demo.APP_NAME, demo.CLASS_NAME),
            (demo.APP_NAME, demo.CLASS_NAME), ("custom-r3-private", "UnderstandingR3Demo")])
        self.assertEqual((demo.APP_NAME, demo.CLASS_NAME), ("ghana-understanding-r2-demo-private", "UnderstandingR2Demo"))
        with self.assertRaisesRegex(ValueError, "separate"):
            demo.routed_events(current, app_name=demo.APP_NAME)

    async def test_r3_shared_history_and_candidate_switch_do_not_leak(self):
        value = manifest()
        calls = []

        async def remote(messages, variant, expected_sha, request_id):
            calls.append((variant, copy.deepcopy(messages)))
            identity = demo.identity(value, expected_sha, variant)
            output = variant + " synthetic test output"
            yield {"type": "start", "protocol": demo.PROTOCOL, "request_id": request_id,
                   "call_id": "fc-fixture", "identity": identity, "messages_sha256": demo.digest(messages)}
            yield {"type": "delta", "request_id": request_id, "index": 0, "text": output}
            yield {"type": "final", "request_id": request_id, "result": {
                "identity": identity, "messages_sha256": demo.digest(messages), "prompt_ids_sha256": "a" * 64,
                "decoding": inference.DECODING, "output": output,
                "output_ids": [1] * 256 if variant == "base" else [1, 2], "seconds": .1,
                "finish_reason": "max_new_tokens" if variant == "base" else "eos", "complete": variant != "base"}}

        async def cancel(_):
            raise AssertionError("Unexpected cancellation")

        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / "r3.sqlite3"

            async def run(state, question, checksum=SHA):
                values = [s async for s in comparison.compare(state, question, value, checksum, inference,
                    store, remote=remote, cancel=cancel)]
                return values[-1]

            first = await run(comparison.new_session("r3_response"), "first")
            self.assertEqual(first["rounds"][0]["status"], "incomplete")
            self.assertFalse(first["rounds"][0]["sides"]["base"]["turn"]["remote_result"]["complete"])
            second = await run(first, "follow up")
            self.assertEqual([v for v, _ in calls], ["base", "r3_response", "base", "r3_response"])
            self.assertEqual(calls[0][1], calls[1][1])
            self.assertEqual(calls[2][1], calls[3][1])
            self.assertEqual(calls[2][1][1]["content"], "r3_response synthetic test output")
            switched = await run(comparison.new_session("r2_meaning"), "new checkpoint")
            self.assertNotEqual(second["id"], switched["id"])
            self.assertEqual(len(calls[-1][1]), 1)
            count = len(calls)
            with self.assertRaisesRegex(ValueError, "Bundle differs"):
                await run(second, "wrong bundle", "b" * 64)
            self.assertEqual(len(calls), count)
            self.assertEqual(comparison.load_session(store, first["id"]), second)


if __name__ == "__main__":
    unittest.main()
