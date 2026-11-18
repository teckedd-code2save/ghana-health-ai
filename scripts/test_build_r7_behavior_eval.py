"""CPU-only synthetic fixtures for freezing inputs, never model-quality results."""
from copy import deepcopy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import build_r7_behavior_eval as builder
import r7_eval_metrics as metrics


def save(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(builder.encoded(value, lines=lines))


class BuilderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pack = self.root / 'r6'
        self.database = self.root / 'regressions.sqlite3'
        self.train = {'id': 'training-1', 'source_id': 'training-source-1',
                      'messages': [{'role': 'user', 'content': 'Unrelated synthetic training fixture text.'},
                                   {'role': 'assistant', 'content': 'Unrelated synthetic answer.'}]}
        self.prepare_pack()
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute('CREATE TABLE r6_sessions (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
            for session_id, count in zip(builder.REGRESSION_SESSIONS, (4, 2)):
                rounds = []
                history = []
                for turn in range(count):
                    messages = history + [{'role': 'user', 'content': 'Synthetic regression question ' + str(turn)}]
                    answer = '1. fixture\n2. fixture\n3. fixture'
                    rounds.append({'id': session_id + '-' + str(turn),
                                   'input_messages': {'r6_response': messages},
                                   'input_messages_sha256': {'r6_response': metrics.digest(messages)},
                                   'sides': {'r6_response': {'turn': {'messages': messages, 'answer': answer,
                                                                   'remote_result': {'output': answer, 'loop_guard': {'trigger': None}}}}}})
                    history = messages + [{'role': 'assistant', 'content': answer}]
                session = {'id': session_id, 'history_policy': 'independent_model_histories_v1',
                           'loop_guard': True, 'rounds': rounds}
                db.execute('INSERT INTO r6_sessions VALUES (?,?)', (session_id, json.dumps(session)))

    def prepare_pack(self):
        source_path = 'datasets/responses/records.jsonl'
        save(self.pack / source_path, [self.train], lines=True)
        save(self.pack / 'source-witnesses.jsonl', [self.train], lines=True)
        artifacts = {name: builder.file_sha(self.pack / name) for name in (source_path, 'source-witnesses.jsonl')}
        save(self.pack / 'manifest.json', {'artifacts': artifacts, 'counts': {'responses': 1},
                                          'dataset_snapshots': [{'records_path': source_path, 'record_count': 1,
                                                                 'records_sha256': artifacts[source_path]}]})
        self.r6_sha = builder.file_sha(self.pack / 'manifest.json')
        save(self.pack / 'PREPARED.json', {'manifest_sha256': self.r6_sha})

    def build(self, name='evaluation'):
        out = self.root / name
        result = builder.build(out, r6_pack=self.pack, expected_r6_manifest_sha256=self.r6_sha,
                               regression_db=self.database)
        return out, result

    def test_reproducible_artifact_bytes_and_no_inference_claim(self):
        first, first_result = self.build('first')
        second, second_result = self.build('second')
        self.assertEqual(first_result, second_result)
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                         {p.name: p.read_bytes() for p in second.iterdir()})
        self.assertFalse(first_result['model_evaluation_executed'])
        manifest = builder.read(first / 'manifest.json')
        self.assertFalse(manifest['training_eligible'])
        self.assertFalse(manifest['native_gold'])
        self.assertFalse(manifest['clinical_gold'])

    def test_exact_group_view_language_probe_counts(self):
        _, result = self.build()
        counts = result['counts']
        self.assertEqual((counts['semantic_task_families'], counts['fresh_source_groups']), (8, 8))
        self.assertEqual((counts['fixed_views'], counts['rollout_scenarios'], counts['rollout_turn_views']), (64, 16, 48))
        self.assertEqual(counts['fresh_derived_test_views'], 112)
        self.assertEqual(counts['fresh_by_language'], {'tw': 56, 'en': 56})
        self.assertEqual((counts['reused_fixed_views'], counts['reused_source_groups']), (5, 2))
        self.assertEqual(counts['fresh_by_probe'], {'fresh': 32, 'correction': 16, 'followup': 32, 'topic_change': 32})

    def test_remote_projections_never_send_expected_or_rubric_fields(self):
        fixed, scenarios = builder.build_views(builder.source_scenarios())
        request = builder.remote_fixed(fixed[0])
        self.assertEqual(set(request), {'id', 'messages'})
        self.assertEqual(request['messages'], fixed[0]['messages'])
        request = builder.remote_rollout(scenarios[0])
        self.assertEqual(set(request), {'id', 'system_message', 'user_turns'})
        self.assertNotIn('expected_per_turn', request)
        request['user_turns'][0] = 'changed'
        self.assertNotEqual(request['user_turns'], scenarios[0]['user_turns'])

    def test_each_language_view_shares_group_and_expected_conditions(self):
        fixed, scenarios = builder.build_views(builder.source_scenarios())
        paired = {}
        for case in builder.all_cases(fixed, scenarios):
            key = (case['source_group_id'], case['mode'], case['probe'])
            paired.setdefault(key, {})[case['input_language']] = case
        for pair in paired.values():
            self.assertEqual(set(pair), {'tw', 'en'})
            self.assertEqual(pair['tw']['constraints'], pair['en']['constraints'])
            self.assertNotEqual(pair['tw']['messages'], pair['en']['messages'])

    def test_correction_history_is_wrong_but_followup_history_is_correct(self):
        fixed, _ = builder.build_views(builder.source_scenarios())
        for case in fixed:
            if case['probe'] not in ('correction', 'followup'):
                continue
            supplied = json.loads(next(m['content'] for m in case['messages'] if m['role'] == 'assistant'))['answer']
            fresh = next(c for c in fixed if c['scenario_id'] == case['scenario_id'] and c['probe'] == 'fresh')
            expected = fresh['constraints'][0]['expected']['value']
            if case['probe'] == 'correction':
                self.assertNotEqual(supplied, expected)
            else:
                self.assertEqual(supplied, expected)

    def test_rollouts_have_no_fixed_assistant_contamination(self):
        _, scenarios = builder.build_views(builder.source_scenarios())
        for scenario in scenarios:
            self.assertEqual(len(scenario['user_turns']), 3)
            for turn, case in enumerate(scenario['expected_per_turn']):
                self.assertEqual(case['turn'], turn)
                self.assertNotIn('assistant', [message['role'] for message in case['messages']])
                self.assertEqual(len(case['messages']), turn + 2)
        scenarios[0]['expected_per_turn'][1]['messages'].append({'role': 'assistant', 'content': 'leaked reply'})
        with self.assertRaisesRegex(ValueError, 'script mismatch'):
            builder.validate_views([], scenarios)

    def test_duplicate_semantic_scenario_with_different_id_is_rejected(self):
        sources = builder.source_scenarios()
        duplicate = deepcopy(sources[0])
        duplicate['id'] = 'renamed-not-independent'
        sources.append(duplicate)
        with self.assertRaisesRegex(ValueError, 'Same semantic scenario'):
            builder.build_views(sources)

    def test_held_content_deduplicates_views_not_source_groups(self):
        fixed, scenarios = builder.build_views(builder.source_scenarios())
        held = builder.held_content(builder.all_cases(fixed, scenarios))
        keys = [(r['source_group_id'], builder.overlap_normalized(r['text'])) for r in held]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len({r['source_group_id'] for r in held}), 8)
        protected_ids = {case_id for record in held for case_id in record['case_ids']}
        self.assertEqual(protected_ids, {c['id'] for c in builder.all_cases(fixed, scenarios)})
        self.assertTrue(all(r['training_eligible'] is False for r in held))

    def test_freshness_rejects_r6_exact_and_normalized_content_overlap(self):
        source = builder.source_scenarios()[0]['facts']['en']
        for text in (source, source.upper().replace(' ', '  ')):
            self.train['messages'][0]['content'] = text
            self.prepare_pack()
            with self.assertRaisesRegex(ValueError, 'overlaps the R6 checkpack'):
                self.build()
            self.assertFalse((self.root / 'evaluation').exists())

    def test_freshness_rejects_r6_group_identity_overlap(self):
        self.train['source_id'] = builder.VERSION + ':categories-01'
        self.prepare_pack()
        with self.assertRaisesRegex(ValueError, 'source_group_identity'):
            self.build()

    def test_changed_r6_source_artifact_and_counts_are_rejected(self):
        (self.pack / 'source-witnesses.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.build()
        self.prepare_pack()
        manifest = builder.read(self.pack / 'manifest.json')
        manifest['counts']['responses'] = 2
        save(self.pack / 'manifest.json', manifest)
        self.r6_sha = builder.file_sha(self.pack / 'manifest.json')
        save(self.pack / 'PREPARED.json', {'manifest_sha256': self.r6_sha})
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            self.build()

    def test_frozen_output_cannot_be_overwritten_and_tampering_fails(self):
        out, result = self.build()
        with self.assertRaisesRegex(ValueError, 'never overwrite'):
            self.build()
        (out / 'cases.jsonl').write_text('[]\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            builder.verify(out, expected_manifest_sha256=result['manifest_sha256'])

    def test_count_tampering_cannot_be_hidden_by_resealing_manifest(self):
        out, _ = self.build()
        manifest = builder.read(out / 'manifest.json')
        manifest['counts']['fresh_source_groups'] = 112
        save(out / 'manifest.json', manifest)
        sha = builder.file_sha(out / 'manifest.json')
        save(out / 'FROZEN.json', {'manifest_sha256': sha})
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            builder.verify(out, expected_manifest_sha256=sha)

    def test_missing_language_and_invalid_expected_answers_rejected(self):
        sources = builder.source_scenarios()
        del sources[0]['queries'][0]['tw']
        with self.assertRaisesRegex(ValueError, 'matched question'):
            builder.build_views(sources)
        sources = builder.source_scenarios()
        sources[0]['wrong'] = sources[0]['answers'][0]
        with self.assertRaisesRegex(ValueError, 'wrong targets'):
            builder.build_views(sources)

    def test_saved_regressions_are_separate_reused_scope_without_native_gold(self):
        cases, provenance = builder.regression_cases(self.database)
        self.assertEqual(provenance['scope'], 'REUSED_development_regression_not_fresh')
        self.assertTrue(all(case['source_group_id'].startswith('reused-r6-session:') for case in cases))
        self.assertTrue(all(case['sources'] == {} for case in cases))
        self.assertEqual(sum(bool(case['constraints']) for case in cases), 2)

    def test_expected_controlled_answers_pass_without_scoring_explanation_semantics(self):
        fixed, _ = builder.build_views(builder.source_scenarios())
        for case in fixed:
            expected = case['constraints'][0]['expected']['value']
            source_id = next(iter(case['sources']))
            row = {'version': metrics.VERSION, 'case': case, 'model_id': 'synthetic-test',
                   'model_sha256': metrics.digest('fixture'), 'protocol_sha256': metrics.digest('fixture-protocol'),
                   'messages': case['messages'], 'response': {'state': 'complete',
                       'output': builder.answer_text(expected, source_id, case['output_language']),
                       'disposition': 'answer', 'executions': [], 'confirmations': []}, 'judgments': []}
            report = metrics.evaluate_rows([row], expected_rows_sha256=metrics.digest([row]))
            self.assertTrue(all(check['passed'] for check in report['rows'][0]['objective'].values()))
            self.assertFalse(report['semantic_quality_established'])


if __name__ == '__main__':
    unittest.main()
