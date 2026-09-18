import copy
import json
import unittest

from summarize_stronger_response import summarize


def fixture():
    schema = {"type": "function", "function": {"name": "search", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
        "required": ["query"], "additionalProperties": False}}}
    return {"run_id": "unit-test", "model": "test", "revision": "pinned", "input_sha256": "test",
        "load_seconds": 1, "elapsed_seconds": 2, "quantization": "fp8", "tensor_parallel_size": 2,
        "inputs": [{"id": "a", "messages": [{"role": "user", "content": "hello"}], "tools": [schema]}],
        "results": [{"id": "a", "thinking": mode, "answer": "Hello", "decoding": {},
            "parsed": {"content": "Hello", "thinking": "PRIVATE_REASONING"},
            "raw_output": "PRIVATE_REASONING<|im_end|>", "finish_reason": "stop", "output_tokens": 10}
            for mode in (False, True)]}


class StrongerSummaryTests(unittest.TestCase):
    def test_reasoning_not_exported_or_semantic_grade_invented(self):
        result = summarize(fixture())
        self.assertNotIn("PRIVATE_REASONING", json.dumps(result))
        self.assertIsNone(result["semantic_grade"])
        self.assertFalse(result["trained_by_project"])

    def test_refuses_missing_duplicate_unknown_and_mismatched_answers(self):
        valid = fixture()
        missing = copy.deepcopy(valid)
        missing["results"].pop()
        duplicate = copy.deepcopy(valid)
        duplicate["results"].append(duplicate["results"][0])
        unknown = copy.deepcopy(valid)
        unknown["results"][0]["id"] = "other"
        answer = copy.deepcopy(valid)
        answer["results"][0]["answer"] = "Different"
        for value in (missing, duplicate, unknown, answer):
            with self.assertRaises(ValueError):
                summarize(value)

    def test_empty_thinking_is_failure_not_a_hidden_answer(self):
        value = fixture()
        value["results"][1].update(answer="", parsed={"thinking": "PRIVATE_REASONING"}, finish_reason="length")
        mode = summarize(value)["modes"]["thinking"]
        self.assertEqual(mode["no_answer_or_tool_ids"], ["a"])
        self.assertEqual(mode["length_limit_ids"], ["a"])

    def test_tool_schema_and_completion_are_checked_without_execution(self):
        value = fixture()
        row = value["results"][0]
        row.update(answer="", parsed={"tool_calls": [{"type": "function", "function": {
            "name": "search", "arguments": {"query": "tomato"}}}]},
            raw_output="<tool_call>...</tool_call><|im_end|>")
        self.assertEqual(summarize(value)["modes"]["non_thinking"]["invalid_tool_requests"], {})
        row["parsed"]["tool_calls"][0]["function"]["arguments"] = {"query": 2}
        self.assertIn("invalid_arguments", summarize(value)["modes"]["non_thinking"]["invalid_tool_requests"]["a"])
        row["finish_reason"] = "length"
        self.assertIn("incomplete_handoff", summarize(value)["modes"]["non_thinking"]["invalid_tool_requests"]["a"])


if __name__ == "__main__":
    unittest.main()
