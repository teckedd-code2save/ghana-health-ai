import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("runtime", Path(__file__).resolve().parents[1] / "modal/response_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class ResponseRuntimeTests(unittest.TestCase):
    def test_reasoning_profile_is_explicit_and_bounded(self):
        normal = runtime.decoding_profile("gemma", "publisher")
        thinking = runtime.decoding_profile("gemma", "gemma-thinking")
        self.assertEqual(normal["generation"], thinking["generation"])
        self.assertFalse(normal["thinking"])
        self.assertTrue(thinking["thinking"])
        self.assertEqual(normal["max_new_tokens"], 512)
        self.assertEqual(thinking["max_new_tokens"], 1536)
        for family, profile in (("qwen", "gemma-thinking"), ("gemma", "qwen-presence"), ("other", "publisher")):
            with self.assertRaises(ValueError):
                runtime.decoding_profile(family, profile)

    def test_presence_penalty_ignores_prompt_and_counts_each_type_once(self):
        import torch
        processor = runtime.presence_processor(2, 1.5)
        scores = torch.tensor([[1., 2., 3., 4.], [-1., -2., -3., -4.]])
        ids = torch.tensor([[0, 1, 2, 2], [0, 1, 3, 2]])
        result = processor(ids, scores)
        self.assertTrue(torch.equal(result, torch.tensor([[1., 2., 1.5, 4.], [-1., -2., -4.5, -5.5]])))
        self.assertTrue(torch.equal(processor(ids[:, :2], scores), scores))
        self.assertTrue(torch.equal(runtime.presence_processor(2, 0)(ids, scores), scores))
        self.assertEqual(scores[0, 2].item(), 3.)

    def call(self, name="calculate_total", arguments=None):
        return {"type": "function", "function": {"name": name, "arguments": arguments or {"unit_price": 7.5, "quantity": 2}}}

    def test_real_decimal_execution_and_tool_result_identity(self):
        import json
        calls, results = runtime.execute_calls([self.call()])
        result = json.loads(results[0]["content"])
        self.assertEqual(result["total"], "15.00")
        self.assertFalse(result["order_placed"])
        self.assertEqual(results[0]["tool_call_id"], calls[0]["id"])

    def test_ungrounded_calculation_does_not_execute(self):
        self.assertEqual(len(runtime.execute_calls([self.call()], [{"role": "user", "content": "7.50 cedis for 1 kg; I need 2 kg."}])[1]), 1)
        for messages in ([{"role": "user", "content": "Help me budget."}],
                         [{"role": "user", "content": "My child feels hot."}, {"role": "assistant", "content": "7.50 and 2"}],
                         [{"role": "user", "content": "-7.50 and 2"}]):
            with self.assertRaises(ValueError):
                runtime.execute_calls([self.call()], messages)

    def test_grounded_numbers_allow_sentence_punctuation_not_partial_numbers(self):
        for text in ("Tomato kilogram baako bo yɛ cedi 7.50. Mepɛ kilogram 2. Fa calculate_total bu ne nyinaa bo kyerɛ me wɔ Twi mu.",
                     "Price: 7.50, quantity: 2.", "Price (7.50); quantity (2)."):
            self.assertEqual(len(runtime.execute_calls([self.call()], [{"role": "user", "content": text}])[1]), 1)
        for text in ("7.50.1 and 2", "7.50e3 and 2", "7.50 and 2,34", "17.50 and 2", "7.50 and -2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                runtime.execute_calls([self.call()], [{"role": "user", "content": text}])
        runtime.execute_calls([self.call(arguments={"unit_price": 1000, "quantity": 2})],
                              [{"role": "user", "content": "The price is 1,000. I need 2."}])

    def test_unknown_consequential_and_malformed_calls_rejected(self):
        from jsonschema import ValidationError
        for call in [None, {}, self.call("place_order"), self.call(arguments={"unit_price": 2}),
                     self.call(arguments={"unit_price": "2", "quantity": 3}),
                     self.call(arguments={"unit_price": True, "quantity": 3}),
                     self.call(arguments={"unit_price": 2, "quantity": 0}),
                     self.call(arguments={"unit_price": 2, "quantity": 3, "command": "bad"}),
                     self.call(arguments={"unit_price": float("nan"), "quantity": 3})]:
            with self.assertRaises((ValueError, ValidationError)):
                runtime.execute_calls([call])

    def test_partial_tool_syntax_cannot_execute(self):
        raw = '<|tool_call>call:calculate_total{unit_price:7.5,quantity:2}<tool_call|><|tool_response>'
        self.assertTrue(runtime.complete_tool_handoff(raw, [10, 50], [self.call()]))
        self.assertFalse(runtime.complete_tool_handoff(raw, [10, 106], [self.call()]))
        self.assertFalse(runtime.complete_tool_handoff(raw.replace("<tool_call|>", ""), [10, 50], [self.call()]))

    def test_model_inputs_cannot_select_paths_or_privileged_messages(self):
        good = [{"role": "user", "content": "Hi"}]
        runtime.validate_input(good, "response_v2", "response_v2_20260910T093637Z", "adapter", "a" * 64)
        for messages, checkpoint in [(good, "../secret"), ([{"role": "system", "content": "Hi"}], "adapter"),
                                      ([{"role": "user", "content": "Hi", "tools": []}], "adapter")]:
            with self.assertRaises(ValueError):
                runtime.validate_input(messages, "response_v2", "response_v2_20260910T093637Z", checkpoint, "a" * 64)
        runtime.validate_input(good, "afrique_v3", "afrique_v3_20260910T102035Z", "adapter", "a" * 64)
        runtime.validate_input(good, "response_v4", "response_v4_20260910T111126Z", "adapter", "a" * 64)
        runtime.validate_input(good, "response_v5", "response_v5_20260910T120000Z", "adapter", "a" * 64)
        runtime.validate_input(good, "afrique_v6", "afrique_v6_20260910T160000Z", "adapter", "a" * 64)
        spec = runtime.model_spec("afrique_v6_20260910T160000Z")
        self.assertEqual(spec["model"], "McGill-NLP/AfriqueQwen3.5-9B-50Langs")
        self.assertEqual(spec["variants"], ("afrique_v6", "afrique9_base"))
        self.assertNotEqual(spec["cache_dir"], spec["tokenizer_cache_dir"])
        with self.assertRaises(ValueError):
            runtime.validate_input(good, "response_v2", "afrique_v3_20260910T102035Z", "adapter", "a" * 64)

    def test_qwen_native_stream_and_real_tool_feedback(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", local_files_only=True)
        tokenizer.response_template = runtime.QWEN_RESPONSE_TEMPLATE
        history = [{"role": "user", "content": "Calculate the price."}]
        prompt = tokenizer.apply_chat_template(history, tools=runtime.TOOLS, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        raw = '<tool_call>\n<function=calculate_total>\n<parameter=unit_price>7.5</parameter>\n<parameter=quantity>2</parameter>\n</function>\n</tool_call><|im_end|>'
        parser = tokenizer.get_response_parser(prefix=prompt, tools=runtime.TOOLS)
        shown = []
        for char in raw:
            shown.extend(runtime.content_chunks(parser.feed(char)))
        parsed, events = parser.finalize()
        shown.extend(runtime.content_chunks(events))
        self.assertEqual("".join(shown).strip(), "")
        self.assertEqual(parsed["tool_calls"][0]["function"]["arguments"], {"unit_price": 7.5, "quantity": 2})
        self.assertTrue(runtime.complete_tool_handoff(raw, [248046], parsed["tool_calls"], "qwen"))
        for partial in (raw.removesuffix("<|im_end|>"), raw.replace("</tool_call>", "")):
            self.assertFalse(runtime.complete_tool_handoff(partial, [248046], parsed["tool_calls"], "qwen"))
        self.assertFalse(runtime.complete_tool_handoff(raw, [99], parsed["tool_calls"], "qwen"))
        calls, replies = runtime.execute_calls(parsed["tool_calls"])
        history.extend([{"role": "assistant", "content": "", "tool_calls": calls}, *replies])
        prompt = tokenizer.apply_chat_template(history, tools=runtime.TOOLS, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        self.assertIn("15.00", prompt)
        parser = tokenizer.get_response_parser(prefix=prompt, tools=runtime.TOOLS)
        shown = []
        for char in "The total is 15 cedis.<|im_end|>":
            shown.extend(runtime.content_chunks(parser.feed(char)))
        parsed, events = parser.finalize()
        shown.extend(runtime.content_chunks(events))
        self.assertEqual("".join(shown), "The total is 15 cedis.")
        self.assertEqual(parsed["content"], "The total is 15 cedis.")

    def test_native_stream_does_not_display_thought_or_tool_syntax(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475", local_files_only=True)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Calculate the price."}],
                    tools=runtime.TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        parser = tokenizer.get_response_parser(prefix=prompt)
        raw = '<|tool_call>call:calculate_total{unit_price:7.5,quantity:2}<tool_call|><|tool_response>'
        shown = []
        for char in raw:
            shown.extend(runtime.content_chunks(parser.feed(char)))
        parsed, events = parser.finalize()
        shown.extend(runtime.content_chunks(events))
        self.assertEqual("".join(shown), "")
        self.assertEqual(parsed["tool_calls"][0]["function"]["arguments"], {"unit_price": 7.5, "quantity": 2})
        calls, replies = runtime.execute_calls(parsed["tool_calls"])
        history = [{"role": "user", "content": "Calculate the price."},
                   {"role": "assistant", "content": "", "tool_calls": calls}, *replies]
        next_prompt = tokenizer.apply_chat_template(history, tools=runtime.TOOLS, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
        self.assertIn("15.00", next_prompt)
        parser = tokenizer.get_response_parser(prefix=next_prompt)
        shown = []
        for char in "The total is 15 cedis.<turn|>":
            shown.extend(runtime.content_chunks(parser.feed(char)))
        parsed, events = parser.finalize()
        shown.extend(runtime.content_chunks(events))
        self.assertEqual("".join(shown), "The total is 15 cedis.")
        self.assertEqual(parsed["content"], "The total is 15 cedis.")

    def test_enabled_reasoning_is_private_and_retained_only_for_tool_turn(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475", local_files_only=True)
        history = [{"role": "user", "content": "Calculate 7.50 times 2."}]
        prompt = tokenizer.apply_chat_template(history, tools=runtime.TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        self.assertIn("<|think|>", prompt)
        parser = tokenizer.get_response_parser(prefix=prompt, tools=runtime.TOOLS)
        raw = '<|channel>thought\nPrivate test reasoning.<channel|><|tool_call>call:calculate_total{unit_price:7.5,quantity:2}<tool_call|><|tool_response>'
        shown = []
        for char in raw:
            shown.extend(runtime.content_chunks(parser.feed(char)))
        parsed, events = parser.finalize()
        shown.extend(runtime.content_chunks(events))
        self.assertEqual("".join(shown), "")
        self.assertEqual(parsed["thinking"], "Private test reasoning.")
        calls, replies = runtime.execute_calls(parsed["tool_calls"], history)
        history.extend([{"role": "assistant", "content": "", "reasoning": parsed["thinking"], "tool_calls": calls}, *replies])
        next_prompt = tokenizer.apply_chat_template(history, tools=runtime.TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        self.assertIn("Private test reasoning.", next_prompt)
        parser = tokenizer.get_response_parser(prefix=next_prompt, tools=runtime.TOOLS)
        shown = []
        for char in '<|channel>thought\nPrivate result check.<channel|>15 cedis.<turn|>':
            shown.extend(runtime.content_chunks(parser.feed(char)))
        parsed, events = parser.finalize()
        shown.extend(runtime.content_chunks(events))
        self.assertEqual("".join(shown), "15 cedis.")


if __name__ == "__main__":
    unittest.main()
