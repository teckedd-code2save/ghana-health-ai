import copy
import hashlib
import json
import unittest

from summarize_oss_response import summarize


class SummaryTests(unittest.TestCase):
    def fixture(self):
        inputs = [{"id": "a", "messages": [{"role": "user", "content": "Hello"}]}]
        return {"inputs": inputs, "input_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "results": [{"id": "a", "answer": "Hello", "parsed": {"content": "Hello", "thinking": "PRIVATE_REASONING"},
                             "finish_reason": "stop", "output_tokens": 5, "raw_output": "PRIVATE_REASONING<|return|>"}],
                **dict.fromkeys(("run_id", "model", "revision", "quantization", "reasoning_effort", "decoding", "load_seconds", "elapsed_seconds"), "test")}

    def test_never_exports_reasoning_or_claims_semantic_score(self):
        result = summarize(self.fixture())
        self.assertNotIn("PRIVATE_REASONING", json.dumps(result))
        self.assertIsNone(result["semantic_accuracy"])
        self.assertFalse(result["trained_by_project"])

    def test_rejects_incomplete_duplicate_or_changed_inputs(self):
        for edit in (lambda r: r["results"].clear(), lambda r: r["results"].append(r["results"][0]),
                     lambda r: r["inputs"][0]["messages"][0].update(content="Changed"),
                     lambda r: r["results"][0].update(answer="Not the parsed final")):
            report = copy.deepcopy(self.fixture())
            edit(report)
            with self.assertRaises(ValueError):
                summarize(report)

    def test_unknown_tool_is_flagged_not_executed(self):
        report = self.fixture()
        report["results"][0]["parsed"]["tool_calls"] = [{"type": "function", "function": {"name": "delete", "arguments": {}}}]
        result = summarize(report)
        self.assertIn("unavailable_function", result["invalid_tool_requests"]["a"])
        self.assertIn("incomplete_tool_handoff", result["invalid_tool_requests"]["a"])
        self.assertFalse(result["tools_executed"])


if __name__ == "__main__":
    unittest.main()
