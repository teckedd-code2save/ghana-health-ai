"""CPU-only sealed fixture tests; no model, service, or registry writes."""
from collections import Counter
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import export_understanding_r5_dataset_view as view
import run_understanding_r5 as r5
from test_run_understanding_r5 import bind_source, fixture, reseal, write


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.current = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.current = [dict(attrs), ""]
            self.scripts.append(self.current)

    def handle_data(self, text):
        if self.current is not None:
            self.current[1] += text

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None


class DatasetViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.pack = fixture(self.root / "pack")
        records_path = self.pack / "datasets/test-v1/records.jsonl"
        records, rows, witnesses = (r5.json_rows(p) for p in
                                   (records_path, self.pack / "rows.jsonl", self.pack / "source-witnesses.jsonl"))
        self.unsafe = '</script><script>globalThis.injected=true</script><img src=x onerror=alert(1)>&\u2028\u2029'
        records[0].update(origin="agent_translated_source", english_messages=[{"role": "user", "content": self.unsafe}],
                          original_ref={"path": "/unavailable/source.json", "row": 12},
                          human_reviewed=False, review={"status": "agent_checked", "reviewer": "test-agent"})
        for row, witness, record in zip(rows, witnesses, records, strict=True):
            row["dataset_ref"]["record_sha256"] = r5.digest(record)
            witness["dataset_ref"] = row["dataset_ref"]
        # A second training prefix shares one registered record, not a second conversation.
        first, witness = copy.deepcopy(rows[0]), copy.deepcopy(witnesses[0])
        first["id"] = witness["id"] = "first-assistant-view"
        witness["messages"] = witness["messages"][:2]
        rows.append(first)
        witnesses.append(witness)
        unused = copy.deepcopy(records[0])
        unused["id"] = "unselected-registry-record"
        records.append(unused)
        write_rows(records_path, records)
        manifest_path = self.pack / "datasets/test-v1/manifest.json"
        manifest = r5.read(manifest_path)
        manifest["record_count"] = len(records)
        write(manifest_path, manifest)
        write_rows(self.pack / "rows.jsonl", rows)
        write_rows(self.pack / "source-witnesses.jsonl", witnesses)
        stages = r5.read(self.pack / "stages.json")
        stages[0]["schedule"].append(4)
        write(self.pack / "stages.json", stages)
        bind_source(self.pack)
        reseal(self.pack)
        self.completed = self.root / "completed"
        self.completed.mkdir()
        self.receipt = self.completed / "consumption.json"
        self.complete()

    def complete(self):
        manifest, rows, stages = r5.verify_pack(self.pack)
        stage = stages[0]
        self.data_sha = r5.sha(self.pack / "manifest.json")
        write(self.completed / "execution.json", {"mode": "train", "data_sha256": self.data_sha})
        execution_sha = r5.sha(self.completed / "execution.json")
        position = {"completed_microbatches": len(stage["schedule"]), "completed_updates": stage["optimizer_updates"],
                    "microbatches_in_update": 0, "exposure": {
                        "row_visits": {str(i): n for i, n in Counter(stage["schedule"]).items()},
                        "input_tokens": stage["sequence_tokens"], "effective_target_tokens": stage["effective_target_tokens"]}}
        checkpoint = {"directory": "/remote/update-00000005", "contract_sha256": "1" * 64,
                      "manifest_sha256": "2" * 64, "state_sha256": "3" * 64}
        run_id = "understanding_20261110T103734Z"
        receipt = {"run_id": run_id, **r5.consumption_receipt(
            rows, stage, manifest["optimization"]["grad_accum"], position,
            snapshots=stage["admitted_dataset_snapshots"], mode="train", execution_sha=execution_sha,
            data_sha=self.data_sha, checkpoint=checkpoint)}
        write(self.receipt, receipt)
        encoded_sha, schedule_sha = view.training_hashes(rows, stage["schedule"])
        terminal = {"stage": "responses", "completion_verified": True, "run_sha256": execution_sha,
                    "data_sha256": self.data_sha, "encoded_rows_sha256": encoded_sha,
                    "schedule_sha256": schedule_sha, "checkpoint": checkpoint["directory"],
                    "checkpoint_run_sha256": checkpoint["contract_sha256"],
                    "checkpoint_completed_updates": stage["optimizer_updates"], "position": position}
        write(self.completed / "status.json", {"state": "completed", "training_completed": True,
            "mode": "train", "data_sha256": self.data_sha, "execution_sha256": execution_sha,
            "consumption_sha256": r5.sha(self.receipt), "run_id": run_id, "stages": [terminal]})

    def load(self):
        return view.load_snapshot(self.pack, self.receipt)[0]

    def rebind_receipt(self):
        status = r5.read(self.completed / "status.json")
        status["consumption_sha256"] = r5.sha(self.receipt)
        write(self.completed / "status.json", status)

    def test_exact_record_view_exposure_and_shifted_supervision_counts(self):
        data = self.load()
        self.assertEqual(len(data["classes"]), 4)
        self.assertEqual(len(data["records"]), 4)
        self.assertEqual(data["totals"]["unique_rows"], 5)
        self.assertEqual(data["totals"]["row_exposures"], 9)
        self.assertEqual(data["totals"]["effective_target_tokens"], 18)
        response = next(r for r in data["records"] if r["class"] == "responses")
        self.assertEqual(response["counts"], {"unique_rows": 2, "row_exposures": 3, "input_tokens": 9,
                                             "effective_target_tokens": 6})
        self.assertEqual([v["message_count"] for v in response["views"]], [4, 2])
        self.assertEqual(response["record"]["english_messages"][0]["content"], self.unsafe)
        self.assertEqual(response["record"]["original_ref"]["path"], "/unavailable/source.json")
        self.assertFalse(response["record"]["human_reviewed"])
        self.assertNotIn("unselected-registry-record", json.dumps(data["records"]))

    def test_all_embedded_records_round_trip_without_script_injection(self):
        data = self.load()
        html = view.render_html(data)
        parser = Scripts()
        parser.feed(html)
        self.assertEqual(len(parser.scripts), 2)
        self.assertEqual(json.loads(parser.scripts[0][1]), data)
        self.assertNotIn("<", parser.scripts[0][1])
        self.assertNotIn("\u2028", parser.scripts[0][1])
        self.assertNotIn("innerHTML", view.SCRIPT)
        self.assertNotIn("fetch(", view.SCRIPT)
        self.assertIn("connect-src 'none'", html)
        self.assertIn("script-src 'sha256-", html)

    def test_tampered_pack_is_rejected(self):
        (self.pack / "rows.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.load()

    def test_wrong_dataset_ref_is_rejected_even_if_resealed(self):
        rows = r5.json_rows(self.pack / "rows.jsonl")
        rows[0]["dataset_ref"]["record_sha256"] = "0" * 64
        write_rows(self.pack / "rows.jsonl", rows)
        reseal(self.pack)
        with self.assertRaisesRegex(ValueError, "dataset reference"):
            self.load()

    def test_rebound_bad_consumption_is_never_silently_dropped(self):
        original = r5.read(self.receipt)
        mutations = [lambda r: r["rows"].pop(), lambda r: r["rows"].append(r["rows"][0]),
                     lambda r: r["rows"][0].update(row_exposures=123),
                     lambda r: r["rows"][0].update(unique_rows=True),
                     lambda r: r["consumed"]["by_class"]["responses"].update(unique_rows=55),
                     lambda r: r.update(training_complete=False),
                     lambda r: r["inventory"].update(archive_counts_included=True)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                receipt = copy.deepcopy(original)
                mutate(receipt)
                write(self.receipt, receipt)
                self.rebind_receipt()
                with self.assertRaisesRegex(ValueError, "Receipt differs"):
                    self.load()

    def test_completion_proof_and_checkpoint_must_match(self):
        path = self.completed / "status.json"
        original = r5.read(path)
        for mutation in (lambda s: s.update(training_completed=False),
                         lambda s: s.update(consumption_sha256="0" * 64),
                         lambda s: s["stages"][0].update(checkpoint="/other"),
                         lambda s: s["stages"][0].update(schedule_sha256="0" * 64)):
            status = copy.deepcopy(original)
            mutation(status)
            write(path, status)
            with self.assertRaises(ValueError):
                self.load()

    def test_missing_or_linked_completion_proof_rejected(self):
        path = self.completed / "status.json"
        path.rename(self.root / "status-copy.json")
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.load()
        path.symlink_to(self.root / "status-copy.json")
        with self.assertRaises(ValueError):
            self.load()

    def test_export_is_private_reproducible_and_does_not_write_inputs(self):
        _, hashes = view.load_snapshot(self.pack, self.receipt)
        first = view.export(self.pack, self.receipt, self.root / "view-1")
        second = view.export(self.pack, self.receipt, self.root / "view-2")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(stat.S_IMODE(Path(first["path"]).stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(Path(first["path"]).parent.stat().st_mode), 0o700)
        view.check_unchanged(hashes)
        with self.assertRaisesRegex(ValueError, "already exists"):
            view.export(self.pack, self.receipt, self.root / "view-1")
        with self.assertRaisesRegex(ValueError, "outside sealed"):
            view.export(self.pack, self.receipt, self.pack / "view")

    def test_change_during_render_prevents_publication(self):
        render = view.render_html
        def changing_render(data):
            (self.pack / "rows.jsonl").write_text("changed")
            return render(data)
        with patch.object(view, "render_html", side_effect=changing_render):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                view.export(self.pack, self.receipt, self.root / "view")
        self.assertFalse((self.root / "view").exists())


if __name__ == "__main__":
    unittest.main()
