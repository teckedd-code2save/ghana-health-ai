import sys
from pathlib import Path
import unittest
from transformers.utils.chat_parsing import ResponseParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal/train"))
from oss_response_core import TEMPLATE


class HarmonyTests(unittest.TestCase):
    def parse(self, raw):
        prefix = "<|start|>assistant<|channel|>final<|message|>Old answer<|end|><|start|>user<|message|>New question<|end|><|start|>assistant"
        parser = ResponseParser(TEMPLATE, prefix=prefix)
        for offset in range(0, len(raw), 7):
            parser.feed(raw[offset:offset+7])
        return parser.finalize()[0]

    def test_reasoning_and_history_do_not_become_answer(self):
        result = self.parse("<|channel|>analysis<|message|>Private reasoning<|end|><|start|>assistant<|channel|>final<|message|>New answer<|return|>")
        self.assertEqual(result["content"], "New answer")
        self.assertEqual(result["thinking"], "Private reasoning")

    def test_both_tool_header_orders(self):
        for header in (" to=functions.web_search<|channel|>commentary json", "<|channel|>commentary to=functions.web_search <|constrain|>json"):
            result = self.parse(header + '<|message|>{"query":"library hours"}<|call|>')
            self.assertEqual(result["tool_calls"][0]["function"], {"name": "web_search", "arguments": {"query": "library hours"}})
            self.assertFalse(result.get("content"))

    def test_exhausted_reasoning_is_not_a_final_answer(self):
        self.assertFalse(self.parse("<|channel|>analysis<|message|>Unfinished").get("content"))


if __name__ == "__main__":
    unittest.main()
