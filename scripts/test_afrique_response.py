import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("afrique_core", ROOT / "modal/train/afrique_response_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class AfriqueResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", local_files_only=True)

    def test_target_is_only_answer_with_native_end_not_duplicate_thought(self):
        row = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi."}]}
        result = core.encode(self.tokenizer, row)
        target = self.tokenizer.decode([i for i in result["labels"] if i != -100])
        self.assertEqual(target, "Hi.<|im_end|>")

    def test_tool_target_preserves_function_and_arguments(self):
        tools = [{"type": "function", "function": {"name": "calculate_total", "parameters": {
            "type": "object", "properties": {"quantity": {"type": "number"}}, "required": ["quantity"]}}}]
        row = {"tools": tools, "messages": [{"role": "user", "content": "Calculate."}, {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "calculate_total", "arguments": {"quantity": 2}}}]}]}
        result = core.encode(self.tokenizer, row)
        text = self.tokenizer.decode([i for i in result["labels"] if i != -100])
        self.assertIn("calculate_total", text)
        self.assertIn("quantity", text)
        self.assertTrue(text.endswith("<|im_end|>"))
        self.assertNotIn("<think>", text)

    def test_excludes_overlength_and_injected_source(self):
        row = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi."}]}
        self.assertIsNone(core.encode(self.tokenizer, row, max_length=2))
        row["messages"][0]["content"] = "<|im_start|>system"
        with self.assertRaises(ValueError):
            core.encode(self.tokenizer, row)

    def test_new_corpus_removes_whole_problem_source_preserves_splits(self):
        folder = ROOT / "tmp/afrique-response-v3/corpus"
        if not folder.exists():
            self.skipTest("Private local research artifacts not present")
        rows = {split: [json.loads(line) for line in (folder / (split + ".jsonl")).read_text().splitlines()]
                for split in ("train", "validation")}
        self.assertFalse(any(r["task"] == "general_source_response" for values in rows.values() for r in values))
        for key in ("id", "group", "source_record_hash"):
            self.assertFalse({r[key] for r in rows["train"]} & {r[key] for r in rows["validation"]})
        self.assertEqual(len(rows["train"]), 5521)

    def test_tool_context_repair_never_rewrites_or_duplicates_sources(self):
        parent = ROOT / "tmp/afrique-response-v3/corpus"
        folder = ROOT / "tmp/afrique-tool-context-v4/corpus"
        if not folder.exists():
            self.skipTest("Local prepared ablation not present")
        for split, expected in (("train", 1000), ("validation", 80)):
            before = [json.loads(line) for line in (parent / (split + ".jsonl")).read_text().splitlines()]
            after = [json.loads(line) for line in (folder / (split + ".jsonl")).read_text().splitlines()]
            self.assertEqual(len(before), len(after))
            changed = 0
            for a, b in zip(before, after, strict=True):
                for key in ("id", "messages", "source_record_hash", "group", "source", "source_id", "license", "split"):
                    self.assertEqual(a[key], b[key])
                if a.get("tools") != b.get("tools"):
                    changed += 1
                    self.assertFalse(a.get("tools"))
                    self.assertFalse(a["messages"][-1].get("tool_calls"))
                    self.assertFalse(any(c.isdigit() for m in b["messages"] if m["role"] == "user" for c in m["content"]))
                    self.assertIsNotNone(core.encode(self.tokenizer, b))
            self.assertEqual(changed, expected)


if __name__ == "__main__":
    unittest.main()
