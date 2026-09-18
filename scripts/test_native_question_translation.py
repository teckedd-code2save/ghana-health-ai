import copy
import hashlib
import unittest
from unittest.mock import patch

import translate_native_questions as client


class NativeQuestionTests(unittest.TestCase):
    def source(self):
        return {"id": "a", "answer_tw": "unaltered", "answer_tw_sha256": hashlib.sha256(b"unaltered").hexdigest(),
                "source": {"source_record_hash": "source-hash", "split": "train", "messages": [{"content": "unaltered"}]},
                "candidates": [{"question_en": "What is needed?", "source_review_candidate": True},
                               {"question_en": "Why?", "source_review_candidate": False}]}

    def test_only_approved_questions_and_no_source_answer_exposure(self):
        rows = client.select_questions([self.source()], set())
        self.assertEqual(rows, [{"id": "a:q0", "source_id": "a", "candidate_index": 0, "question_en": "What is needed?"}])
        for change in ("flag", "split", "answer", "duplicate"):
            row = copy.deepcopy(self.source())
            if change == "split":
                row["source"]["split"] = "validation"
            if change == "answer":
                row["answer_tw"] = "rewritten"
            with self.assertRaises(ValueError):
                client.select_questions([row, row] if change == "duplicate" else [row], {"source-hash"} if change == "flag" else set())

    def test_input_integrity_and_complete_coverage(self):
        inputs = [{"id": "a", "task": "translation", "prompt": "English: What?\nTwi:"}]
        result = {"run_id": "run", "model": client.MODEL, "revision": client.MODEL_REVISIONS["afrique"],
                  "inputs_sha256": client.digest(inputs), "results": [{"id": "a", "task": "translation", "prediction": "Dɛn?"}]}
        receipt = {"run_id": "run", "state": "completed", "inputs_sha256": client.digest(inputs), "results_sha256": client.digest(result)}
        files = {"forward.inputs.json": inputs, "forward.receipt.json": receipt, "forward.results.json": result}
        with patch.object(client, "load", side_effect=files.__getitem__):
            self.assertEqual(client.completed("forward")["a"]["prediction"], "Dɛn?")
            result["results"][0]["prediction"] = "Changed"
            with self.assertRaises(ValueError):
                client.completed("forward")
            receipt["results_sha256"] = client.digest(result)
            result["results"].append(result["results"][0])
            receipt["results_sha256"] = client.digest(result)
            with self.assertRaises(ValueError):
                client.completed("forward")

    def test_edited_snapshot_does_not_reuse_translation(self):
        files = {"questions.json": [{"id": "a", "question_en": "What?"}]}
        files["manifest.json"] = {"snapshots": {"questions.json": client.digest(files["questions.json"])}}
        with patch.object(client, "load", side_effect=files.__getitem__):
            self.assertEqual(client.snapshot("questions.json")[0]["question_en"], "What?")
            files["questions.json"][0]["question_en"] = "When?"
            with self.assertRaises(ValueError):
                client.snapshot("questions.json")

    def test_original_answer_meaning_is_not_leaked_to_back_translator(self):
        snapshots = {"demonstrations.json": [
            {"language": "en", "messages": [{"content": "Translate from Twi to English.\n\nDɛn?"}, {"content": "What?"}]}],
            "questions.json": [{"id": "a:q0", "question_en": "ORIGINAL QUESTION"}],
            "sources.json": [{"id": "a", "answer_tw": "NATIVE ANSWER", "answer_en": "HIDDEN REFERENCE"}]}
        with patch.object(client, "snapshot", side_effect=snapshots.__getitem__), patch.object(client, "completed", return_value={"a:q0": {"prediction": "TWI QUESTION"}}):
            result = client.translation_inputs("back")
            self.assertEqual([r["id"] for r in result], ["a:q0", "a:answer"])
            self.assertIn("TWI QUESTION", result[0]["prompt"])
            self.assertIn("NATIVE ANSWER", result[1]["prompt"])
            self.assertNotIn("ORIGINAL QUESTION", str(result))
            self.assertNotIn("HIDDEN REFERENCE", str(result))

    def test_export_preserves_original_and_never_claims_verified_translation(self):
        source = {**self.source(), "answer_en": "ORIGINAL ANSWER"}
        before = copy.deepcopy(source)
        snapshots = {"sources.json": [source], "questions.json": [{"id": "a:q0", "source_id": "a", "question_en": "What is needed?"}]}
        results = {"forward": {"a:q0": {"prediction": "TWI QUESTION", "finish_reason": "stop"}},
                   "back": {"a:q0": {"prediction": "What is needed?", "finish_reason": "stop"},
                            "a:answer": {"prediction": "TRANSLATED ANSWER", "finish_reason": "stop"}}}
        saved = {"manifest.json": {}}
        with patch.object(client, "snapshot", side_effect=snapshots.__getitem__), patch.object(client, "completed", side_effect=results.__getitem__), patch.object(client, "save", side_effect=saved.__setitem__), patch.object(client, "load", side_effect=saved.__getitem__), patch.object(client, "question_findings", return_value={}), patch("builtins.print"):
            client.export()
        row = saved["review.json"][0]
        self.assertEqual(source, before)
        self.assertEqual(row["answer_tw"], before["answer_tw"])
        self.assertEqual(row["answer_en"], "ORIGINAL ANSWER")
        self.assertEqual(row["answer_back_translation_en"], "TRANSLATED ANSWER")
        self.assertFalse(row["translated_question_candidates"][0]["translation_verified"])
        self.assertFalse(row["standalone_conversation_eligible"])
        self.assertFalse(row["train_eligible"])

    def test_findings_cannot_be_reused_for_another_run(self):
        import json
        findings = {"forward_result_logical_sha256": "original", "findings": [{"id": "a", "issue": "Changed quantity"}]}
        with patch.object(client.Path, "read_text", return_value=json.dumps(findings)), patch.object(client, "load", return_value={"results_sha256": "original"}):
            self.assertEqual(client.question_findings({"a"})["a"]["issue"], "Changed quantity")
            with self.assertRaises(ValueError):
                client.question_findings({"b"})
        with patch.object(client.Path, "read_text", return_value=json.dumps(findings)), patch.object(client, "load", return_value={"results_sha256": "different"}):
            with self.assertRaises(ValueError):
                client.question_findings({"a"})


if __name__ == "__main__":
    unittest.main()
