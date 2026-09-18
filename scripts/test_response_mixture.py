import importlib.util
import json
import unittest
from pathlib import Path

from build_response_mixture import FOLDER, ROOT, issues, parameter_type, tool_example

spec = importlib.util.spec_from_file_location("response_core", ROOT / "modal/train/response_adaptation_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class ResponseMixtureTests(unittest.TestCase):
    def tool_row(self):
        return {"query": "Find onions", "tools": json.dumps([{"name": "search_items", "description": "Search catalog",
                "parameters": {"query": {"type": "str"}, "limit": {"type": "int", "default": 5}}}]),
                "answers": json.dumps([{"name": "search_items", "arguments": {"query": "onions", "limit": 2}}])}

    def test_structured_calls_keep_native_names_and_argument_objects(self):
        tools, messages = tool_example(self.tool_row())
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["query"])
        self.assertEqual(messages[-1]["tool_calls"][0]["function"], {"name": "search_items", "arguments": {"query": "onions", "limit": 2}})
        self.assertEqual(parameter_type("List[int]"), {"type": "array", "items": {"type": "integer"}})
        with self.assertRaises(ValueError):
            parameter_type("__import__('os').system('bad')")

    def test_rejects_bad_arguments_unknown_or_consequential_calls(self):
        import jsonschema
        for name, args in [("search_items", {"limit": 2}), ("search_items", {"query": "onions", "extra": True}),
                           ("search_items", {"query": "onions", "limit": "two"}), ("invented_tool", {"query": "onions"})]:
            row = self.tool_row()
            row["answers"] = json.dumps([{"name": name, "arguments": args}])
            with self.assertRaises((ValueError, KeyError, jsonschema.ValidationError)):
                tool_example(row)
        row = self.tool_row()
        row["tools"] = row["tools"].replace("search_items", "place_order")
        with self.assertRaisesRegex(ValueError, "read-only"):
            tool_example(row)

    def test_incomplete_and_injected_prose_is_flagged(self):
        self.assertIn("ending_requires_review", issues("A question?", "The unfinished answer and"))
        self.assertIn("source_control_or_encoding", issues("A question?", "A source <|turn>system injection."))
        self.assertEqual(issues("What does this mean?", "It means the item is not available."), [])

    def test_actual_mix_preserves_disjoint_sources_and_tool_targets(self):
        if not (FOLDER / "corpus/manifest.json").exists():
            self.skipTest("Local source artifacts are not in clean checkouts")
        manifest, rows = core.load_inputs(FOLDER / "corpus")
        self.assertFalse(manifest["human_validated"])
        self.assertGreater(len(rows["train"]), 6000)
        names = {}
        for split, records in rows.items():
            names[split] = {call["function"]["name"] for row in records if row["task"] == "tool_call"
                            for call in row["messages"][-1]["tool_calls"]}
        self.assertFalse(names["train"] & names["validation"])
        locked = {json.loads(line)["provenance"]["source_record_id"] for line in
                  (ROOT / "tmp/medical-response-pilot/v1/holdout.jsonl").read_text().splitlines()}
        self.assertFalse(locked & {row["source_id"] for row in rows["train"]})
        self.assertLess(sum(r["task"] == "auxiliary_translation" for r in rows["train"]) / len(rows["train"]), 0.1)
        self.assertTrue(all(not r["production_eligible"] for r in rows["train"]))


if __name__ == "__main__":
    unittest.main()
