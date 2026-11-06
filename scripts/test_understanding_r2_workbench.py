import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import understanding_r2_workbench as ui
from test_understanding_r2_demo import manifest, SHA
import r2_inference


class WorkbenchTests(unittest.TestCase):
    def test_layout_ids_are_unique_load_targets_exist_and_startup_is_offline(self):
        with tempfile.TemporaryDirectory() as folder:
            remote = Mock(side_effect=AssertionError("No remote startup call"))
            with patch.object(ui.demo, "bound_bundle", return_value=(manifest(), r2_inference)):
                app = ui.build_app(Path(folder), SHA, Path(folder) / "chat.sqlite3", remote=remote)
            remote.assert_not_called()
            self.assertFalse(app.is_running)
            cfg = app.get_config_file()
            components = cfg["components"]
            tabs = [c["props"]["label"] for c in components if c["type"] == "tabitem"]
            self.assertEqual(tabs, ["Compare", "Chat"])
            controls = [c for c in components if c.get("props", {}).get("label") == "Candidate checkpoint"]
            self.assertEqual(controls[0]["props"]["value"], "r2_meaning")
            self.assertNotIn("base", [v for _, v in controls[0]["props"]["choices"]])
            def ids(node):
                return [node["id"]] + [i for child in node.get("children", []) for i in ids(child)]
            layout_ids = ids(cfg["layout"])
            self.assertEqual(len(layout_ids), len(set(layout_ids)))
            allowed = set(layout_ids) | {app._id}
            for dep in cfg["dependencies"]:
                for target, event in dep["targets"]:
                    if event == "load":
                        self.assertIn(target, allowed)

    def test_metrics_escape_error_and_do_not_invent_timing(self):
        markup = ui.side_metrics({"status": "error", "error": "<script>oops</script>", "turn": None})
        self.assertNotIn("<script>", markup)
        self.assertNotIn("First text", markup)

    def test_transcript_keeps_raw_answer_and_separate_sides(self):
        state = {"rounds": [{"question": "Question", "sides": {
            "base": {"turn": {"answer": "Base answer"}}, "candidate": {"turn": {"answer": "R2 answer"}}}}]}
        self.assertEqual(ui.transcript(state, "base")[-1]["content"], "Base answer")
        self.assertEqual(ui.transcript(state, "candidate")[-1]["content"], "R2 answer")


if __name__ == "__main__":
    unittest.main()
