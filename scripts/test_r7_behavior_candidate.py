"""Synthetic R7 candidate execution and paired reports; no Modal or model calls."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import inspect
from pathlib import Path
import tempfile
from types import FunctionType
import unittest
from unittest.mock import patch

import prepare_understanding_r7_inference as builder
import r7_inference as inference
import run_r7_behavior_baseline as baseline_launcher
import run_r7_behavior_candidate as launcher
import score_r7_behavior_baseline as baseline_scorer
import score_r7_behavior_candidate as scorer
from test_prepare_understanding_r7_inference import fixture
from test_score_r7_behavior_baseline import FixtureRuntime


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        readback, run = fixture(cls.root)
        receipt = cls.root / 'receipt.json'
        adapters = {'r6_response': {'directory': inference.R6_DIRECTORY, 'files': {
            'adapter_config.json': inference.R6_CONFIG, 'adapter_model.safetensors': inference.R6_WEIGHTS}},
            'r7_response': {'directory': '/artifacts/understanding-adaptation-r7/' + run.name + '/responses-adapter',
                            'files': {name: inference.sha(run / 'responses-adapter' / name)
                                      for name in ('adapter_config.json', 'adapter_model.safetensors')}}}
        inference.base.save(readback / 'adapter-readback.json', {key: {**value, 'verified': True,
            'bytes_read': {name: 1 for name in value['files']}, 'method': 'SYNTHETIC TEST ONLY'} for key, value in adapters.items()})
        cls.bundle = cls.root / 'bundle'
        prepared = builder.prepare(builder.ROOT / 'tmp/r6-demo/bundle-v1', receipt, readback, cls.bundle,
                                   receipt_sha=inference.sha(receipt))
        cls.bundle_sha = prepared['manifest_sha256']
        cls.evaluation = builder.ROOT / 'tmp/r7-20261117/behavior-eval-v1'
        cls.eval_sha = '8029612cea3110d31d9b8a1f717e0f43a38148c613c739341ae46f50d3a68e13'
        cls.package = cls.root / 'execution'
        execution = launcher.prepare(cls.package, cls.evaluation, cls.eval_sha, cls.bundle,
                                     'r7-candidate-synthetic', expected_bundle_sha=cls.bundle_sha)
        cls.execution_sha = execution['execution_sha256']
        cls.worker, cls.execution, cls.plan, cls.slots = launcher.verify_package(cls.package, cls.execution_sha)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)

    def test_same_117_gold_free_inputs_and_matched_raw_protocol(self):
        old = baseline_launcher.load_worker().read(builder.ROOT / 'tmp/r7-20261117/r6-behavior-execution-v4/inputs.json')
        self.assertEqual(self.plan, old)
        self.assertEqual(len(self.slots), 117)
        self.assertEqual(self.execution['binding']['variant'], 'r7_response')
        self.assertEqual(self.execution['binding']['protocol_sha256'], self.worker.MATCHED_PROTOCOL_SHA)
        self.assertNotEqual(self.execution['app_name'], baseline_launcher.load_worker().APP_NAME)

    def test_altered_prompt_or_decoding_cannot_be_a_matched_candidate(self):
        manifest = self.worker.verify_bundle(self.bundle, self.bundle_sha)
        changed_plan = deepcopy(self.plan)
        changed_plan['cases'][0]['messages'][-1]['content'] += ' altered'
        with self.assertRaisesRegex(ValueError, 'unchanged frozen'):
            self.worker.binding_for('r7-candidate-test', changed_plan, manifest, self.bundle_sha,
                                    self.execution['binding']['worker_sha256'], self.eval_sha)
        manifest['decoding']['repetition_penalty'] = 1.1
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            self.worker.binding_for('r7-candidate-test', self.plan, manifest, self.bundle_sha,
                                    self.execution['binding']['worker_sha256'], self.eval_sha)

    def test_r6_bundle_cannot_be_relabelled_as_r7(self):
        with self.assertRaisesRegex(ValueError, 'R7 inference bundle'):
            self.worker.verify_bundle(builder.ROOT / 'tmp/r6-demo/bundle-v1', inference.R6_BUNDLE_SHA)

    def test_cli_prepare_requires_and_passes_independent_bundle_hash(self):
        path = self.out / 'cli-package'
        with redirect_stdout(io.StringIO()):
            result = launcher.main(['prepare', '--package', str(path), '--evaluation', str(self.evaluation),
                '--evaluation-sha', self.eval_sha, '--bundle', str(self.bundle), '--expected-bundle-sha', self.bundle_sha,
                '--run-id', 'r7-candidate-cli-fixture'])
        self.assertFalse(result['gpu_launched'])
        self.assertEqual(result['counts']['expected_slots'], 117)

    def test_candidate_failures_block_only_own_rollout_and_score_all_slots(self):
        raw = self.out / 'progress'
        report = self.worker.evaluate_local(raw, self.plan, self.execution['binding'],
                                            lambda: FixtureRuntime(fail_indices=(0, 69)))
        converted = scorer.convert(self.package, self.execution_sha, raw / 'progress.json', report['snapshot_sha256'])
        self.assertEqual(converted['summary']['counts']['states']['error'], 2)
        self.assertEqual(converted['summary']['counts']['states']['blocked'], 2)
        self.assertEqual(len(converted['rows']['fresh']), 112)
        self.assertEqual(len(converted['rows']['reused']), 5)
        self.assertEqual({row['model_id'] for row in converted['rows']['fresh']}, {'r7_response'})
        old_weights = inference.R6_WEIGHTS
        self.assertNotEqual(converted['summary']['model_identity']['adapter_weights_sha256'], old_weights)

    def test_launch_without_authorization_never_initializes_backend(self):
        with patch.object(launcher, 'ModalBackend') as backend:
            with self.assertRaisesRegex(ValueError, 'authorize-gpu'):
                launcher.launch(self.package, self.execution_sha, self.out / 'receipt.json')
            backend.assert_not_called()

    def test_real_modal_serialization_does_not_capture_unhydrated_claims(self):
        from modal._serialization import serialize
        with patch.object(self.worker, 'require_launch_python'):
            app, function, claims = self.worker.create_app(self.bundle, self.bundle_sha)
        self.assertEqual(app.name, 'ghana-r7-behavior-candidate-private')
        self.assertFalse(claims.is_hydrated)
        raw = inspect.unwrap(function.get_raw_f())
        self.assertNotIn('claims', raw.__code__.co_freevars)
        def cell(value):
            return (lambda: value).__closure__[0]
        closure = tuple(cell('mounted-volume-placeholder') if name == 'volume' else value
                        for name, value in zip(raw.__code__.co_freevars, raw.__closure__))
        probe = FunctionType(raw.__code__, raw.__globals__, raw.__name__, raw.__defaults__, closure)
        self.assertGreater(len(serialize(probe)), 100)

    def test_paired_report_preserves_scope_modes_groups_and_failed_denominators(self):
        raw = self.out / 'r7-raw'
        report = self.worker.evaluate_local(raw, self.plan, self.execution['binding'], lambda: FixtureRuntime(fail_indices=(69,)))
        candidate_path = self.out / 'r7-score'
        candidate = scorer.score(self.package, self.execution_sha, raw / 'progress.json', report['snapshot_sha256'], candidate_path)
        before_package = self.out / 'r6-execution'
        before_prepared = baseline_launcher.prepare(before_package, self.evaluation, self.eval_sha,
            builder.ROOT / 'tmp/r6-demo/bundle-v1', 'r7-baseline-paired-fixture')
        before_worker, execution, plan, _ = baseline_launcher.verify_package(before_package, before_prepared['execution_sha256'])
        before_raw = self.out / 'r6-raw'
        before = before_worker.evaluate_local(before_raw, plan, execution['binding'], lambda: FixtureRuntime())
        baseline_path = self.out / 'r6-score'
        baseline = baseline_scorer.score(before_package, before_prepared['execution_sha256'], before_raw / 'progress.json',
                                          before['snapshot_sha256'], baseline_path)
        paired = scorer.compare_with_r6(baseline_path, baseline['manifest_sha256'], candidate_path,
                                       candidate['manifest_sha256'], resamples=100)
        modes = paired['scopes']['fresh']['answer']['by_mode']
        self.assertEqual(modes['fixed_shared_history']['matched_rows'], 64)
        self.assertEqual(modes['independent_rollout']['matched_rows'], 48)
        self.assertEqual(modes['independent_rollout']['unavailable_candidate_in_matched'], 3)
        self.assertEqual(modes['independent_rollout']['group_count'], 8)
        self.assertEqual(modes['fixed_shared_history']['uncertainty']['resamples'], 100)
        self.assertIn('reused', paired['scopes'])
        self.assertFalse(paired['semantic_quality_established'])


if __name__ == '__main__':
    unittest.main()
