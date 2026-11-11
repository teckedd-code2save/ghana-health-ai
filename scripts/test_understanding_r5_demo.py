"""Offline private R5 actor tests with sealed helpers and a CPU-only fake runtime."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import prepare_understanding_r5_inference as builder
import r5_inference as r5
from test_prepare_understanding_r5_inference import fixture, HELPERS


class ActorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        parent, checksum, readback, _ = fixture(self.root)
        self.result = builder.prepare(parent, checksum, self.root / "receipt.json", self.root / "terminal.json",
                                      readback, self.root / "bundle", helpers_root=HELPERS, include_pass1=True)
        self.bundle = self.root / "bundle"
        self.checksum = self.result["manifest_sha256"]
        self.manifest = r5.load_bundle(self.bundle, self.checksum)

    def test_actor_is_separate_private_readonly_and_bounded(self):
        source = (builder.ROOT / "modal/research_understanding_r5_demo.py").read_text()
        tree = ast.parse(source)
        actor = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertEqual(actor.name, "UnderstandingR5Demo")
        options = {k.arg: k.value for k in actor.decorator_list[0].keywords}
        for key, value in {"min_containers": 0, "max_containers": 1, "retries": 0,
                           "include_source": False, "gpu": "H100", "timeout": 300,
                           "startup_timeout": 600, "cpu": 4, "memory": 49152}.items():
            self.assertEqual(ast.literal_eval(options[key]), value)
        self.assertEqual(source.count(".read_only()"), 2)
        for forbidden in (".deploy(", ".spawn(", ".remote(", "fastapi_endpoint", "web_endpoint"):
            self.assertNotIn(forbidden, source)
        self.assertIn('from guarded_r3_runtime import reply', source)
        self.assertIn('sys.path.insert(0, str(BUNDLE / "code"))', source)
        self.assertNotIn('/scripts/understanding_r2_demo.py', source)

    def test_variant_identity_and_unadapted_base_use_sealed_runtime(self):
        runtime = r5.Runtime.__new__(r5.Runtime)
        runtime.manifest, runtime.manifest_sha = self.manifest, self.checksum
        for variant, spec in self.manifest["variants"].items():
            self.assertEqual(runtime.identity(variant)["adapter_weights_sha256"],
                             None if spec is None else spec["files"]["adapter_model.safetensors"])
        for variant in ("r2_meaning", "r3_response"):
            with self.assertRaisesRegex(ValueError, "Unknown"):
                runtime.identity(variant)
        disabled = Mock()
        runtime.loaded, runtime.model = {"r5_response"}, SimpleNamespace(disable_adapter=disabled)
        runtime.select("base")
        disabled.assert_called_once_with()

    def test_import_and_raw_guarded_streams_never_load_gpu_or_deploy(self):
        import os
        image = Mock()
        for method in ("pip_install", "env", "add_local_dir", "add_local_file"):
            getattr(image, method).return_value = image
        app = SimpleNamespace(cls=lambda **options: lambda cls: cls)
        volume = SimpleNamespace(read_only=lambda: object())
        fake_modal = SimpleNamespace(
            is_local=lambda: True, App=lambda name: app,
            Image=SimpleNamespace(debian_slim=lambda **kw: image),
            Volume=SimpleNamespace(from_name=lambda *args, **kw: volume),
            enter=lambda: lambda fn: fn, method=lambda **kw: lambda fn: fn,
            current_function_call_id=lambda: "fc-offline-test")
        source = builder.ROOT / "modal/research_understanding_r5_demo.py"
        spec = importlib.util.spec_from_file_location("_r5_actor_offline_test", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": fake_modal}), patch.object(sys, "path", list(sys.path)), \
                patch.object(sys, "dont_write_bytecode", False), \
                patch.dict(os.environ, GHA_R5_DEMO_BUNDLE=str(self.bundle), GHA_R5_DEMO_SHA=self.checksum):
            # Test isolation only: production refuses helpers imported from a different tree.
            for name in ("understanding_r2_demo", "response_loop_guard", "guarded_r3_runtime"):
                sys.modules.pop(name, None)
            exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
            demo = sys.modules["understanding_r2_demo"]
            self.assertEqual(Path(demo.__file__).resolve(), self.bundle / "code/understanding_r2_demo.py")
            self.assertFalse(list(self.bundle.rglob("__pycache__")))
            actor = module.UnderstandingR5Demo()
            actor.setup()
            self.assertIsNone(actor.runtime)
            messages = [{"role": "user", "content": "Synthetic stream test"}]
            text = "one two three four five six seven eight " * 5

            class FakeRuntime:
                def identity(inner, variant):
                    return demo.identity(self.manifest, self.checksum, variant)

                def reply(inner, received, variant, on_text, cancelled):
                    self.assertEqual(received, messages)
                    on_text(text)
                    stopped = cancelled()
                    return {"identity": inner.identity(variant), "messages_sha256": demo.digest(received),
                            "prompt_ids_sha256": "a" * 64, "decoding": self.manifest["decoding"],
                            "output_ids": [1, 248046], "output": text, "seconds": .1,
                            "finish_reason": "cancelled" if stopped else "eos", "complete": not stopped}
            actor.runtime = FakeRuntime()
            guard = Mock(side_effect=lambda runtime, *args, **kw: runtime.reply(*args, **kw))
            with patch.dict(sys.modules, {"guarded_r3_runtime": SimpleNamespace(reply=guard)}):
                for variant in self.manifest["variants"]:
                    raw = list(actor.stream(messages, variant, self.checksum, "a" * 32))
                    guarded = list(actor.stream_guarded(messages, variant, self.checksum, "b" * 32))
                    self.assertNotIn("loop_guard", raw[0])
                    self.assertEqual(guarded[0]["loop_guard"], "repeat-block-v1")
                    left, right = raw[-1]["result"], guarded[-1]["result"]
                    self.assertEqual(left["decoding"], right["decoding"])
                    self.assertEqual(left["output"], right["output"])
                    self.assertEqual(left["finish_reason"], "eos")
                    self.assertEqual(right["finish_reason"], "cancelled")
                    self.assertFalse(right["complete"])
                    self.assertIsNotNone(right["loop_guard"]["trigger"])
                self.assertEqual(guard.call_count, len(self.manifest["variants"]))
            with self.assertRaisesRegex(ValueError, "bundle/variant"):
                list(actor.stream(messages, "r2_meaning", self.checksum, "c" * 32))
            with self.assertRaisesRegex(ValueError, "bundle/variant"):
                list(actor.stream(messages, "base", "0" * 64, "c" * 32))


if __name__ == "__main__":
    unittest.main()
