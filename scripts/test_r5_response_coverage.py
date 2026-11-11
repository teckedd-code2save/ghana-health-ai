import unittest

from prepare_understanding_r5 import response_coverage


def row(identity, origin, size, *, source="shared", new=False, collection="responses"):
    return {"id": identity, "source_id": source, "preparation_origin": origin,
            "collection": collection, "new_source_conversation": new,
            "input_ids": [1] * (size + 1), "labels": [-100] + [1] * size}


class ResponseCoverageTests(unittest.TestCase):
    def test_exposures_do_not_inflate_source_or_view_counts(self):
        rows = [row("source", "new_reviewed_translation", 10, new=True),
                row("repair", "new_reviewed_translation", 5, source="project"),
                row("spelling", "new_spelling_augmentation", 10),
                row("english", "reviewed_english_source", 50, collection="english_retention")]
        result = response_coverage(rows, [0, 1, 2, 3, 0])
        self.assertEqual(result["scheduled_response_target_tokens"], 35)
        groups = result["groups"]
        self.assertEqual(groups["new_source_backed_conversation"]["target_tokens"], 20)
        self.assertEqual(groups["new_source_backed_conversation"]["distinct_views"], 1)
        self.assertEqual(groups["new_source_backed_conversation"]["source_ids"], 1)
        self.assertEqual(groups["new_project_authored"]["target_tokens"], 5)
        self.assertNotIn("reviewed_english_source", groups)
        self.assertAlmostEqual(sum(g["response_token_share"] for g in groups.values()), 1)
        self.assertFalse(result["source_counts_additive"])

    def test_empty_response_inventory_does_not_invent_a_percentage(self):
        self.assertEqual(response_coverage([], [])["groups"], {})


if __name__ == "__main__":
    unittest.main()
