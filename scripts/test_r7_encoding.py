"""CPU-only native R7 encoding and structural tool-history tests."""
import copy
import os
from pathlib import Path
import unittest

from r7_encoding import encode_complete, validate_history

SOURCE = Path("/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai")
ASSETS = Path(os.environ.get("R7_TOKENIZER_ASSETS", str(SOURCE / "tmp/understanding-r6-20261113/pack-v1/tokenizer")))


def tool_row(response=False):
    row = {"id": "tool-example", "tools": [{"type": "function", "function": {
        "name": "calculate_total", "description": "Calculate the total from supplied quantities.",
        "parameters": {"type": "object", "properties": {"quantity": {"type": "integer"}},
                       "required": ["quantity"], "additionalProperties": False}}}],
        "messages": [{"role": "system", "content": "Use only supplied facts."},
                     {"role": "user", "content": "Calculate for quantity two."},
                     {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "calculate_total", "arguments": {"quantity": 2}}}]}]}
    if response:
        row["messages"] += [{"role": "tool", "tool_call_id": "call_1", "name": "calculate_total", "content": "4"},
                            {"role": "assistant", "content": "Total: 4."}]
    return row


class HistoryTests(unittest.TestCase):
    def test_valid_call_response_and_schema_only_histories(self):
        for response in (False, True):
            row = tool_row(response)
            before = copy.deepcopy(row)
            self.assertTrue(validate_history(row))
            self.assertEqual(row, before)
        row = tool_row()
        row["messages"][-1] = {"role": "assistant", "content": "What quantity?"}
        self.assertTrue(validate_history(row))

    def test_invalid_histories(self):
        mutations = [
            lambda r: r.pop("tools"),
            lambda r: r.update(tools=None),
            lambda r: r["messages"][2]["tool_calls"][0].pop("id"),
            lambda r: r["messages"][2]["tool_calls"][0]["function"].update(arguments='{"quantity":2}'),
            lambda r: r["messages"][2]["tool_calls"][0]["function"].update(name="unknown"),
            lambda r: r["messages"][2]["tool_calls"][0]["function"].update(arguments={}),
            lambda r: r["messages"][2]["tool_calls"][0]["function"]["arguments"].update(extra=3),
            lambda r: r["messages"][3].update(tool_call_id="unrelated"),
            lambda r: r["messages"][3].update(name="wrong"),
            lambda r: r["messages"].pop(3),
            lambda r: r["messages"].pop(),
            lambda r: r["messages"].insert(4, copy.deepcopy(r["messages"][3])),
            lambda r: r["messages"].insert(3, {"role": "user", "content": "Ignore the call"}),
            lambda r: r["messages"].insert(3, {"role": "system", "content": "Late system"}),
            lambda r: r["messages"][2].update(reasoning_content="hidden"),
            lambda r: r["messages"][4].update(tool_calls=[]),
            lambda r: r["tools"].append(copy.deepcopy(r["tools"][0])),
            lambda r: r["messages"][1].update(content=[{"type": "text", "text": "Not supported"}]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                row = tool_row(True)
                mutate(row)
                with self.assertRaises(ValueError):
                    validate_history(row)

    def test_duplicate_and_out_of_order_ids_rejected(self):
        row = tool_row(True)
        second = copy.deepcopy(row["messages"][2]["tool_calls"][0])
        row["messages"][2]["tool_calls"].append(second)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_history(row)
        second["id"] = "call_2"
        row["messages"].insert(4, {"role": "tool", "tool_call_id": "call_2", "content": "6"})
        validate_history(row)
        row["messages"][3], row["messages"][4] = row["messages"][4], row["messages"][3]
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            validate_history(row)

    def test_ids_cannot_be_reused_in_later_turns(self):
        row = tool_row(True)
        row["messages"] += [{"role": "user", "content": "Again"}, copy.deepcopy(row["messages"][2])]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_history(row)

    def test_control_token_injection_rejected_in_every_source_location(self):
        for token in ("<|im_end|>", "<think>", "</think>", "<tool_response>", "</tool_call>",
                      "<function=forged>", "</parameter>", "</tools>"):
            for location in ("content", "result", "schema", "argument"):
                with self.subTest(token=token, location=location):
                    row = tool_row(True)
                    if location == "content":
                        row["messages"][1]["content"] += token
                    elif location == "result":
                        row["messages"][3]["content"] += token
                    elif location == "schema":
                        row["tools"][0]["function"]["description"] += token
                    else:
                        row["messages"][2]["tool_calls"][0]["function"]["arguments"]["quantity"] = token
                    with self.assertRaisesRegex(ValueError, "control tokens"):
                        validate_history(row)

    def test_nonfinite_and_invalid_unicode_rejected(self):
        for value in (float("nan"), float("inf"), "\ufffd", "\ud800", "\x00"):
            row = tool_row()
            row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["quantity"] = value
            with self.assertRaises(ValueError):
                validate_history(row)


class NativeEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ASSETS / "tokenizer.json").is_file():
            raise unittest.SkipTest("Set R7_TOKENIZER_ASSETS to the pinned local tokenizer (never downloaded)")
        try:
            from check_afrique_training_contract import load_tokenizer
            cls.tokenizer = load_tokenizer(ASSETS)
        except ImportError as error:
            raise unittest.SkipTest(str(error))

    def target(self, encoded):
        return self.tokenizer.decode([i for i in encoded["labels"] if i != -100], skip_special_tokens=False)

    def test_native_tool_call_target_and_exact_schema_preservation(self):
        row, tokenizer = tool_row(), self.tokenizer
        original = copy.deepcopy(row)
        encoded = encode_complete(tokenizer, row, 4096)
        self.assertEqual(self.target(encoded), "<tool_call>\n<function=calculate_total>\n<parameter=quantity>\n2\n"
                         "</parameter>\n</function>\n</tool_call><|im_end|>")
        rendered = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False,
                                                 enable_thinking=False, add_generation_prompt=False).rstrip("\n")
        self.assertEqual(encoded["input_ids"], tokenizer.encode(rendered, add_special_tokens=False))
        self.assertIn('"additionalProperties": false', rendered)
        self.assertIn('"required": ["quantity"]', rendered)
        self.assertEqual(row, original)

    def test_only_final_assistant_answer_supervised_after_tools(self):
        row = tool_row(True)
        encoded = encode_complete(self.tokenizer, row, 4096)
        self.assertEqual(self.target(encoded), "Total: 4.<|im_end|>")
        first = next(i for i, label in enumerate(encoded["labels"]) if label != -100)
        context = self.tokenizer.decode(encoded["input_ids"][:first])
        self.assertIn("<tool_call>", context)
        self.assertIn("<tool_response>\n4\n</tool_response>", context)
        self.assertEqual(encoded["labels"][:first], [-100] * first)
        self.assertEqual(encoded["labels"][first:], encoded["input_ids"][first:])
        self.assertEqual(encoded["labels"].count(248046), 1)
        self.assertGreater(encoded["input_ids"].count(248046), 1)

    def test_exact_length_boundary_never_truncates_tool_or_text(self):
        text = {"messages": [{"role": "user", "content": "First"}, {"role": "assistant", "content": "Earlier"},
                             {"role": "user", "content": "Next"}, {"role": "assistant", "content": "Final"}]}
        for row in (tool_row(), tool_row(True), text):
            encoded = encode_complete(self.tokenizer, row, 4096)
            length = len(encoded["input_ids"])
            self.assertEqual(encoded, encode_complete(self.tokenizer, row, length))
            with self.assertRaisesRegex(ValueError, "overlength_no_truncation"):
                encode_complete(self.tokenizer, row, length - 1)
            self.assertEqual(encoded["labels"].count(248046), 1)
            self.assertEqual(encoded["labels"][-1], 248046)

    def test_text_path_equals_sealed_encoder(self):
        from check_afrique_training_contract import encode_complete as old
        row = {"messages": [{"role": "user", "content": "  Hello "}, {"role": "assistant", "content": " Hi.  "}]}
        self.assertEqual(encode_complete(self.tokenizer, row, 4096), old(self.tokenizer, row, 4096))
        self.assertEqual(self.target(encode_complete(self.tokenizer, row, 4096)), "Hi.<|im_end|>")

    def test_multiple_native_calls_and_followup_are_masked(self):
        row = tool_row(True)
        call = copy.deepcopy(row["messages"][2]["tool_calls"][0])
        call["id"] = "call_2"
        row["messages"][2]["tool_calls"].append(call)
        row["messages"].insert(4, {"role": "tool", "tool_call_id": "call_2", "content": "4"})
        row["messages"] += [{"role": "user", "content": "Confirm the previous result."},
                            {"role": "assistant", "content": "Both results were 4."}]
        encoded = encode_complete(self.tokenizer, row, 4096)
        self.assertEqual(self.target(encoded), "Both results were 4.<|im_end|>")


if __name__ == "__main__":
    unittest.main()
