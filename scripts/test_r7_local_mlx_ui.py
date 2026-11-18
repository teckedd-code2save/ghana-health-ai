"""UI construction only; no model is loaded or fake chat served."""
import tempfile
import threading
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from r7_local_mlx import serve


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Requires optional local Gradio runtime")
class UIContract(unittest.TestCase):
    def test_local_comparison_constructs_without_cloud_launch(self):
        import gradio as gr
        captured = {}

        def capture(app, **kwargs):
            captured.update(config=app.get_config_file(), launch=kwargs)

        runtime = SimpleNamespace(identity={"mode": "test-construction-only"}, stop=threading.Event())
        with tempfile.TemporaryDirectory() as directory, patch.object(gr.Blocks, "launch", capture):
            serve(runtime, Path(directory), 7871)
        self.assertEqual(captured["launch"]["server_name"], "127.0.0.1")
        self.assertFalse(captured["launch"]["share"])
        config = captured["config"]
        self.assertFalse(config["analytics_enabled"])
        labels = [c["props"].get("label") for c in config["components"] if c["type"] == "chatbot"]
        self.assertEqual(labels, ["R7", "R6"])
        guard = next(c for c in config["components"] if c["props"].get("label") == "Repetition guard")
        self.assertTrue(guard["props"]["value"])


if __name__ == "__main__":
    unittest.main()
