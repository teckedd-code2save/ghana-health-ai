"""Synthetic raw checkpoints against the real frozen input schema; no inference/API calls."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import run_r7_behavior_baseline as launcher
import score_r7_behavior_baseline as scorer


class FixtureRuntime:
    def __init__(self, *, fail_indices=(), empty_indices=(), incomplete_indices=(), clock=None):
        self.calls = 0
        self.fail_indices, self.empty_indices, self.incomplete_indices = fail_indices, empty_indices, incomplete_indices
        self.clock = clock

    def reply(self, messages, on_text):
        index = self.calls
        self.calls += 1
        if self.clock is not None:
            self.clock[0] += 2
        if index in self.fail_indices:
            on_text('Synthetic partial error output')
            raise RuntimeError('Synthetic fixture failure')
        if index in self.empty_indices:
            return {'output': '', 'complete': False}
        return {'output': '{"answer":[],"citations":[],"explanation":"Synthetic output, not a model prediction."}',
                'complete': index not in self.incomplete_indices}


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.shared.cleanup)
        cls.package = Path(cls.shared.name) / 'package'
        root = launcher.ROOT
        prepared = launcher.prepare(cls.package, root / 'tmp/r7-20261117/behavior-eval-v1',
                                    '8029612cea3110d31d9b8a1f717e0f43a38148c613c739341ae46f50d3a68e13',
                                    root / 'tmp/r6-demo/bundle-v1', 'r7-baseline-scoring-fixture')
        cls.execution_sha = prepared['execution_sha256']
        cls.worker, cls.execution, cls.plan, cls.slots = launcher.verify_package(cls.package, cls.execution_sha)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.progress_path = self.root / 'checkpoint/progress.json'

    def checkpoint(self, runtime=None, **kwargs):
        runtime = runtime or FixtureRuntime()
        return self.worker.evaluate_local(self.progress_path.parent, self.plan, self.execution['binding'], lambda: runtime, **kwargs)

    def convert(self, report, **kwargs):
        return scorer.convert(self.package, self.execution_sha, self.progress_path, report['snapshot_sha256'], **kwargs)

    def review_fixture(self, report):
        converted = self.convert(report)
        decisions = {'version': scorer.REVIEW_VERSION, 'execution_sha256': self.execution_sha,
                     'progress_snapshot_sha256': report['snapshot_sha256'],
                     'reviewer': {'kind': 'model', 'id': 'synthetic-test-judge', 'version': 'fixture-v1',
                                  'version_disclosed': True, 'native_twi_reviewer': False},
                     'policy': 'Synthetic unknown judgments only; not a model evaluation.', 'records': []}
        for rows in converted['rows'].values():
            for row in rows:
                decisions['records'].append({'case_id': row['case']['id'], 'whole_answer_label': 'unknown',
                    'full_answer_reviewed': False, 'native_review_required': row['case']['output_language'] == 'tw',
                    'negation_refusal': 'unknown', 'reason': 'Unreviewed synthetic fixture.',
                    'labels': {dimension: 'unknown' for dimension in row['case']['rubrics']}})
        return decisions

    def annotated(self, report, decisions):
        path = self.root / 'annotations.json'
        self.worker.save(path, decisions)
        return self.convert(report, annotations=path, expected_annotations_sha=scorer.file_sha(path))

    def test_all_117_rows_have_actual_fresh_and_reused_denominators(self):
        report = self.checkpoint()
        converted = self.convert(report)
        self.assertEqual({scope: len(rows) for scope, rows in converted['rows'].items()}, {'fresh': 112, 'reused': 5})
        self.assertEqual(converted['summary']['scopes']['fresh']['source_groups'], 8)
        self.assertEqual(converted['summary']['scopes']['reused']['source_groups'], 2)
        self.assertEqual(converted['summary']['scopes']['fresh']['by_mode'],
                         {'fixed_shared_history': 64, 'independent_rollout': 48})
        self.assertEqual(converted['summary']['counts']['expected_slots'], 117)
        self.assertEqual(sum(converted['summary']['scopes']['fresh']['states'].values()), 112)

    def test_errors_invalid_incomplete_and_blocked_are_not_dropped(self):
        # Fixed requests occupy 69 slots; the next inference starts rollout zero.
        report = self.checkpoint(FixtureRuntime(fail_indices=(0, 69), empty_indices=(1,), incomplete_indices=(2,)))
        converted = self.convert(report)
        rows = [row for scoped in converted['rows'].values() for row in scoped]
        states = {state: sum(row['response']['state'] == state for row in rows) for state in scorer.metrics.STATES}
        self.assertEqual(states, report['counts']['states'])
        self.assertEqual((states['error'], states['invalid'], states['incomplete'], states['blocked']), (2, 1, 1, 2))
        self.assertEqual(len(rows), 117)
        failed = converted['rows']['fresh'][0]
        self.assertEqual(failed['response']['output'], 'Synthetic partial error output')
        self.assertEqual(converted['reports']['fresh']['rows'][0]['objective']['answer']['passed'], False)

    def test_partial_checkpoint_emits_every_pending_slot(self):
        clock = [0]
        report = self.checkpoint(FixtureRuntime(clock=clock), seconds=1, clock=lambda: clock[0])
        converted = self.convert(report)
        self.assertEqual(report['state'], 'paused')
        self.assertEqual(converted['summary']['scopes']['fresh']['states']['pending'], 111)
        self.assertEqual(converted['summary']['scopes']['reused']['states']['pending'], 5)
        self.assertEqual(converted['summary']['counts']['finalized_slots'], 1)

    def test_loading_error_preserves_all_pending_denominators(self):
        def fail():
            raise RuntimeError('Synthetic loading error')
        report = self.worker.evaluate_local(self.progress_path.parent, self.plan, self.execution['binding'], fail)
        converted = self.convert(report)
        self.assertEqual(converted['summary']['raw_progress_state'], 'error')
        self.assertEqual(converted['summary']['scopes']['fresh']['states']['pending'], 112)
        self.assertEqual(converted['summary']['scopes']['reused']['states']['pending'], 5)

    def test_remote_turn_ids_map_to_case_ids_without_losing_histories(self):
        converted = self.convert(self.checkpoint())
        rollout = next(row for row in converted['rows']['fresh'] if row['case']['mode'] == 'independent_rollout' and row['case']['turn'] == 1)
        source = next(source for source in converted['row_sources'] if source['case_id'] == rollout['case']['id'])
        self.assertTrue(source['remote_slot_id'].endswith(':turn-1'))
        self.assertTrue(rollout['case']['id'].endswith(':followup'))
        self.assertEqual([m['role'] for m in rollout['messages']], ['system', 'user', 'assistant', 'user'])

    def test_evidence_and_completion_never_supply_semantic_gold(self):
        converted = self.convert(self.checkpoint())
        self.assertEqual(converted['summary']['evidence']['worker_report_verified_slots'], 117)
        self.assertFalse(converted['summary']['semantic_quality_established'])
        self.assertFalse(converted['summary']['clinical_certification'])
        self.assertFalse(converted['summary']['evidence']['quality_inferred_from_evidence_verification'])
        for rows in converted['rows'].values():
            self.assertTrue(all(row['judgments'] == [] and row['response']['disposition'] == 'unknown' for row in rows))
        self.assertTrue(all(not row['objective'].get('answer', {}).get('passed', False)
                            for row in converted['reports']['fresh']['rows']))

    def test_sealed_report_recomputes_and_preserves_exact_raw_bytes(self):
        report = self.checkpoint()
        first, second = self.root / 'score-a', self.root / 'score-b'
        result = scorer.score(self.package, self.execution_sha, self.progress_path, report['snapshot_sha256'], first)
        again = scorer.score(self.package, self.execution_sha, self.progress_path, report['snapshot_sha256'], second)
        self.assertEqual(result['manifest_sha256'], again['manifest_sha256'])
        self.assertEqual((first / 'raw-progress.json').read_bytes(), self.progress_path.read_bytes())
        self.assertEqual(scorer.verify(first, expected_manifest_sha256=result['manifest_sha256']), result)
        with self.assertRaisesRegex(ValueError, 'new score directory'):
            scorer.score(self.package, self.execution_sha, self.progress_path, report['snapshot_sha256'], first)

    def test_changed_snapshot_rejected_even_if_new_self_hash_is_supplied_in_file(self):
        report = self.checkpoint()
        altered = deepcopy(report)
        altered['results'][0]['output'] = 'changed'
        altered['snapshot_sha256'] = self.worker.digest({k: v for k, v in altered.items() if k != 'snapshot_sha256'})
        self.worker.save(self.progress_path, altered)
        with self.assertRaisesRegex(ValueError, 'independent progress snapshot'):
            self.convert(report)

    def test_duplicate_raw_results_are_rejected_after_resealing(self):
        report = self.checkpoint()
        report['results'].append(deepcopy(report['results'][0]))
        report['snapshot_sha256'] = self.worker.digest({k: v for k, v in report.items() if k != 'snapshot_sha256'})
        self.worker.save(self.progress_path, report)
        with self.assertRaisesRegex(ValueError, 'Duplicate progress'):
            self.convert(report)

    def test_mapping_cannot_move_fresh_rows_into_reused_scope(self):
        report = self.checkpoint()
        package = self.root / 'altered-package'
        shutil.copytree(self.package, package)
        slots = self.worker.read(package / 'slots.json')
        slots[0]['scope'] = 'reused'
        self.worker.save(package / 'slots.json', slots)
        execution = self.worker.read(package / 'execution.json')
        execution['files']['slots.json'] = scorer.file_sha(package / 'slots.json')
        self.worker.save(package / 'execution.json', execution)
        checksum = scorer.file_sha(package / 'execution.json')
        self.worker.save(package / 'PREPARED.json', {'execution_sha256': checksum})
        with self.assertRaisesRegex(ValueError, 'mapping differs'):
            scorer.convert(package, checksum, self.progress_path, report['snapshot_sha256'])

    def test_tampered_sealed_metric_counts_and_quality_claims_are_rejected(self):
        report = self.checkpoint()
        out = self.root / 'scored'
        result = scorer.score(self.package, self.execution_sha, self.progress_path, report['snapshot_sha256'], out)
        summary = self.worker.read(out / 'summary.json')
        summary['semantic_quality_established'] = True
        self.worker.save(out / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            scorer.verify(out, expected_manifest_sha256=result['manifest_sha256'])
        manifest = self.worker.read(out / 'manifest.json')
        manifest['artifacts']['summary.json'] = scorer.file_sha(out / 'summary.json')
        self.worker.save(out / 'manifest.json', manifest)
        checksum = scorer.file_sha(out / 'manifest.json')
        self.worker.save(out / 'FROZEN.json', {'manifest_sha256': checksum})
        with self.assertRaisesRegex(ValueError, 'Unsupported quality claim'):
            scorer.verify(out, expected_manifest_sha256=checksum)

    def test_annotations_store_exact_inputs_output_hashes_and_references_not_gold(self):
        report = self.checkpoint()
        converted = self.annotated(report, self.review_fixture(report))
        self.assertEqual(len(converted['review']['records']), 117)
        record = converted['review']['records'][0]
        raw = report['results'][record['raw_ref']['record_index']]
        self.assertEqual(record['raw_output_sha256'], scorer.hashlib.sha256(raw['output'].encode()).hexdigest())
        self.assertEqual(record['raw_ref']['slot_id'], raw['id'])
        judge = record['judgments'][0]['judge']
        self.assertEqual(judge['input_sha256'], scorer.metrics.digest(judge['input']))
        self.assertEqual(judge['version'], 'fixture-v1')
        self.assertEqual(converted['summary']['semantic_judgments'], 'attributed_agent_not_gold')
        self.assertFalse(converted['review']['model_judgments_are_gold'])
        fresh = next(g for g in converted['review_summary']['groups'] if g['labels'] == {'scope': 'fresh'})
        self.assertEqual(fresh['expected_rows'], 112)
        self.assertEqual(fresh['source_groups'], 8)
        self.assertEqual(fresh['whole_answer_agent_labels']['unknown'], 112)

    def test_review_missing_duplicate_wrong_snapshot_and_unreported_version_rejected(self):
        report = self.checkpoint()
        for mutation in ('missing', 'duplicate', 'snapshot', 'version'):
            value = self.review_fixture(report)
            if mutation == 'missing':
                value['records'].pop()
            elif mutation == 'duplicate':
                value['records'].append(deepcopy(value['records'][0]))
            elif mutation == 'snapshot':
                value['progress_snapshot_sha256'] = '0' * 64
            else:
                value['reviewer']['version_disclosed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.annotated(report, value)

    def test_twi_native_unknown_cannot_become_implicit_language_or_whole_answer_pass(self):
        report = self.checkpoint()
        for mutation in ('language', 'whole_answer'):
            value = self.review_fixture(report)
            item = value['records'][0]
            item.update(full_answer_reviewed=True, negation_refusal='clear')
            if mutation == 'language':
                item['labels']['language_adherence'] = 'pass'
            else:
                item['whole_answer_label'] = 'pass'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.annotated(report, value)

    def test_negation_refusal_or_unreviewed_whole_answer_cannot_pass(self):
        report = self.checkpoint()
        for flag, reviewed in (('present', True), ('unknown', True), ('clear', False)):
            value = self.review_fixture(report)
            item = next(item for item in value['records'] if ':en:' in item['case_id'])
            item.update(whole_answer_label='pass', negation_refusal=flag, full_answer_reviewed=reviewed)
            with self.subTest(flag=flag, reviewed=reviewed), self.assertRaises(ValueError):
                self.annotated(report, value)

    def test_changed_annotation_file_requires_independent_hash(self):
        report = self.checkpoint()
        value = self.review_fixture(report)
        path = self.root / 'review.json'
        self.worker.save(path, value)
        checksum = scorer.file_sha(path)
        value['records'][0]['reason'] = 'Changed review'
        self.worker.save(path, value)
        with self.assertRaisesRegex(ValueError, 'annotation file hash'):
            self.convert(report, annotations=path, expected_annotations_sha=checksum)

    def test_controlled_symbols_not_keyword_hit_or_scalar_shape_relaxation(self):
        converted = self.convert(self.checkpoint())
        row = deepcopy(converted['rows']['fresh'][0])
        row['response']['output'] = '{"answer":[],"explanation":"Not triangle, square, circle."}'
        self.assertFalse(scorer.controlled_diagnostics(row, {})['strict_answer_match'])
        row['case']['constraints'][0]['expected']['value'] = [6]
        row['response']['output'] = '{"answer":6,"explanation":"6"}'
        result = scorer.controlled_diagnostics(row, {})
        self.assertFalse(result['strict_answer_match'])
        self.assertTrue(result['scalar_singleton_equivalent_not_a_pass'])
        row['response']['state'] = 'error'
        row['case']['probe'] = 'topic_change'
        self.assertFalse(scorer.controlled_diagnostics(row, {})['constraint_change_exact'])

    def test_review_rejects_row_output_not_matching_raw_reference(self):
        report = self.checkpoint()
        converted = self.convert(report)
        decisions = self.review_fixture(report)
        converted['rows']['fresh'][0]['response']['output'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'referenced raw output'):
            scorer.bind_review(converted['rows'], converted['row_sources'], report, decisions, self.execution_sha)

    def test_annotated_score_recomputes_counts_and_preserves_judgments(self):
        report = self.checkpoint()
        decisions = self.review_fixture(report)
        path = self.root / 'annotations.json'
        self.worker.save(path, decisions)
        out = self.root / 'annotated-score'
        result = scorer.score(self.package, self.execution_sha, self.progress_path, report['snapshot_sha256'], out,
                              annotations=path, expected_annotations_sha=scorer.file_sha(path))
        self.assertEqual(scorer.verify(out, expected_manifest_sha256=result['manifest_sha256']), result)
        self.assertTrue((out / 'agent-review.json').is_file())
        self.assertTrue((out / 'review-summary.json').is_file())

    def test_explicit_profiles_expand_to_identical_bound_reviews(self):
        report = self.checkpoint()
        decisions = self.review_fixture(report)
        original = self.annotated(report, decisions)
        compact = deepcopy(decisions)
        compact['profiles'] = {}
        for record in compact['records']:
            name = 'native' if record['native_review_required'] else 'other'
            compact['profiles'][name] = {key: value for key, value in record.items() if key not in ('case_id', 'labels')}
            compact['profiles'][name].update(default_label='unknown', labels={})
        compact['records'] = [{'case_id': record['case_id'],
                               'profile': 'native' if record['native_review_required'] else 'other'}
                              for record in decisions['records']]
        self.assertEqual(self.annotated(report, compact)['review'], original['review'])
        compact['records'][0]['profile'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'Unknown review profile'):
            self.annotated(report, compact)

    def test_pending_change_probes_stay_in_denominator_and_known_wrong_change_is_not_uptake(self):
        clock = [0]
        report = self.checkpoint(FixtureRuntime(clock=clock), seconds=1, clock=lambda: clock[0])
        converted = self.annotated(report, self.review_fixture(report))
        fresh = next(g for g in converted['review_summary']['groups'] if g['labels'] == {'scope': 'fresh'})
        self.assertEqual(fresh['controlled_changes']['correction_uptake_exact'], {'expected': 16, 'passed': 0})
        self.assertEqual(fresh['controlled_changes']['constraint_change_exact'], {'expected': 64, 'passed': 0})
        row = deepcopy(next(row for row in converted['rows']['fresh'] if ':quantity-01:tw:' in row['case']['id']
                            and row['case']['probe'] == 'correction'))
        row['response'].update(state='complete', output='{"answer":[13],"explanation":"9"}')
        diagnostic = scorer.controlled_diagnostics(row, {})
        self.assertTrue(diagnostic['injected_wrong_prior'])
        self.assertFalse(diagnostic['copied_injected_wrong_answer'])
        self.assertFalse(diagnostic['correction_uptake_exact'])


if __name__ == '__main__':
    unittest.main()
