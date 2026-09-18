import copy
import hashlib
import json
import unittest

from run_stronger_response_comparison import validate_result


class ReceiptTests(unittest.TestCase):
    def fixture(self):
        inputs = [{"id": "a", "messages": [{"role": "user", "content": "Hello"}]}]
        digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        receipt = {"run_id": "run", "model": "model", "revision": "sha", "rows": 1, "modes": ["one"], "input_sha256": digest}
        report = {"run_id": "run", "model": "model", "revision": "sha", "input_sha256": digest, "inputs": inputs, "results": [{"id": "a"}]}
        return receipt, report

    def test_single_and_paired_complete_results(self):
        receipt, report = self.fixture()
        validate_result(report, receipt)
        receipt["modes"] = ["off", "on"]
        report["results"] = [{"id": "a", "thinking": False}, {"id": "a", "thinking": True}]
        validate_result(report, receipt)

    def test_wrong_identity_missing_changed_or_duplicate_rejected(self):
        original, initial = self.fixture()
        for edit in (lambda r: r.update(model="wrong"), lambda r: r["results"].clear(),
                     lambda r: r["results"].append(r["results"][0]),
                     lambda r: r["inputs"][0]["messages"][0].update(content="Changed")):
            report = copy.deepcopy(initial)
            edit(report)
            with self.assertRaises(ValueError):
                validate_result(report, original)
