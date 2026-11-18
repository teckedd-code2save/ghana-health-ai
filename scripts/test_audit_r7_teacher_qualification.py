import copy
import unittest

from audit_r7_teacher_qualification import FINDINGS, audit
from r7_teacher_controls import controls


class QualificationReviewTests(unittest.TestCase):
    def report(self, profile="gemma31"):
        return {"status": "complete", "profile": profile, "run_id": "test", "model": "test",
            "revision": "pinned", "requests": [{"id": r["id"], "messages": r["messages"]} for r in controls()],
            "results": [{"id": r["id"], "output": FINDINGS[profile].get(r["id"], ("unreviewed",))[0]} for r in controls()]}

    def test_no_implicit_passing_or_training(self):
        for profile in FINDINGS:
            result = audit(self.report(profile))
            self.assertFalse(result["training_eligible"])
            self.assertEqual(len(result["findings"]), 3)
            self.assertEqual(result["unlisted_controls"], "unadjudicated_not_implicitly_passed")

    def test_changed_output_or_input_invalidates_review(self):
        original = self.report()
        for change in (lambda r: r["results"][1].update(output="Different result"),
                       lambda r: r["requests"][1]["messages"][-1].update(content="Different source"),
                       lambda r: r.update(status="running"),
                       lambda r: r["results"].append(r["results"][0]),
                       lambda r: r["results"].pop()):
            report = copy.deepcopy(original)
            change(report)
            with self.assertRaises(ValueError):
                audit(report)


if __name__ == "__main__":
    unittest.main()
