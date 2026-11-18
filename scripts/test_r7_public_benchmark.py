from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import r7_public_benchmark_protocol as p
import r7_public_benchmark_launch as launch
import r7_public_benchmark_protection as protection
import r7_public_benchmark_protection_audit as protection_audit

ROOT = Path(__file__).resolve().parents[1]
worker = p.import_file(ROOT / 'modal/train/r7_public_benchmark.py')


def fixtures():
    rows = [p.protocol.normalize('mmlu', {'question': 'Question ' + str(i), 'choices': ['first', 'second'], 'answer': i % 2}, i)
            for i in range(3)]
    sib = p.protocol.normalize('sib200_twi', {'text': 'A classification case', 'category': 'health'}, 0)
    inputs = {'mmlu': rows, 'sib200_twi': [sib]}
    binding = {'version': p.VERSION, 'run_id': 'r7-public-test-0001', 'model': 'foundation', 'revision': 'f' * 40,
               'base_sha256': 'b' * 64, 'bundle_sha256': 'a' * 64, 'runtime': {'torch': 'pinned'},
               'tokenizer_files': {'tokenizer/tokenizer.json': 'c' * 64},
               'adapters': {v: {'files': {'adapter_model.safetensors': str(i) * 64}} for i, v in enumerate(p.VARIANTS)}}
    common = {'runtime': binding['runtime'], 'tokenizer_files': binding['tokenizer_files'],
              'key_mapping': {r'^model\.language_model\.': 'model.'}, 'base_weights_verified': True,
              'no_meta_parameters': True, 'strict_loading': {'strict_text_weights_complete': True, 'tensors': 10},
              'attention': 'sdpa', 'dtype': 'torch.bfloat16', 'device': 'cuda', 'tf32': False,
              'kernel_bindings': {'chunk': 'fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule',
                                  'recurrent': 'fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule'}}
    proofs = {v: {'identity': {'manifest_sha256': binding['bundle_sha256'], 'variant': v, 'model': binding['model'],
                             'revision': binding['revision'], 'base_weights_sha256': binding['base_sha256'],
                             'adapter_weights_sha256': binding['adapters'][v]['files']['adapter_model.safetensors']},
                  'common': deepcopy(common), 'adapter_values_exact': True} for v in p.VARIANTS}
    return inputs, binding, proofs


class FakeRuntime:
    def __init__(self, proofs, fail=None):
        self.proofs, self.fail, self.calls = proofs, fail, []

    def select(self, variant):
        self.variant = variant
        return self.proofs[variant]

    def score(self, row):
        self.calls.append((self.variant, row['id']))
        if self.fail == row['id']:
            raise RuntimeError('simulated model failure')
        base = {'prompt_ids_sha256': p.digest(row['prompt']), 'prompt_tokens': 5}
        if row['task'] == 'mcq':
            return {**base, 'prediction': 'A', 'answer_token_ids': {'A': [1], 'B': [2]}, 'scores': {'A': -1., 'B': -2.}}
        return {**base, 'prediction': None, 'raw_output': 'health or sports', 'output_ids': [10, 11]}


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return self.value


class ProtocolTests(unittest.TestCase):
    def test_single_position_only_presented_candidates(self):
        calls = []
        def forward(ids, targets, **kwargs):
            calls.append((ids, targets, kwargs))
            return [-3., -1., -2.]
        result = p.conditional_scores([10, 20], {'A': [1], 'B': [2], 'C': [3]}, forward)
        self.assertEqual(p.protocol.choose(result, list('ABC')), 'B')
        self.assertEqual(calls, [([10, 20], [1, 2, 3], {'same_position': True})])

    def test_multitoken_conditional_all_positions_and_no_length_normalization(self):
        calls = []
        def forward(ids, targets, **kwargs):
            calls.append((ids, targets, kwargs))
            return [-1.] if targets == [1] else [-.75, -.75]
        result = p.conditional_scores([10, 20], {'A': [1], 'B': [2, 3]}, forward)
        self.assertEqual(result, {'A': -1., 'B': -1.5})
        self.assertEqual(calls[1], ([10, 20, 2], [2, 3], {'same_position': False}))

    def test_nonfinite_positive_or_wrong_length_scores_fail(self):
        for values in ([math.nan], [math.inf], [.1], [], [-1., -2.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                p.conditional_scores([1], {'A': [2]}, lambda *a, **k: values)

    def test_exact_frozen_scorer_and_strict_sib(self):
        self.assertEqual(p.sha(ROOT / 'scripts/r6_benchmark_protocol.py'), p.PROTOCOL_SHA)
        for value in ('health and sports', 'The answer is health', 'unhealthy', ''):
            self.assertIsNone(p.protocol.classify_output(value))
        self.assertEqual(p.protocol.classify_output('Category: **Health**.'), 'health')

    def test_boundary_merge_rejected(self):
        tokenizer = SimpleNamespace(encode=lambda text, **_: [1] if text == 'P' else [2, 3])
        with self.assertRaises(ValueError):
            p.protocol.answer_tokens(tokenizer, 'P', ['A', 'B'])

    def test_missing_rows_never_reduce_expected_denominator(self):
        inputs, _, _ = fixtures()
        report = p.report(inputs, [])
        self.assertEqual(report['total']['r6_response']['expected'], 4)
        self.assertEqual(report['total']['r6_response']['pending'], 4)
        self.assertIsNone(report['total']['paired']['primary_delta_points'])
        self.assertEqual(report['tasks']['sib200_twi']['labels']['health']['paired']['expected'], 1)
        self.assertIsNone(report['tasks']['sib200_twi']['labels']['sports']['r6_response']['final_accuracy'])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'
        self.inputs, self.binding, self.proofs = fixtures()

    def run_engine(self, runtime=None, **kwargs):
        runtime = runtime or FakeRuntime(self.proofs)
        report = worker.evaluate_local(self.root, self.inputs, self.binding, lambda: runtime, **kwargs)
        return runtime, report

    def test_full_run_preserves_cases_and_strict_invalid_counts(self):
        runtime, progress = self.run_engine()
        self.assertEqual(progress['state'], 'complete')
        self.assertEqual(len(runtime.calls), 8)
        self.assertEqual(len(list((self.root / 'cases').rglob('*.json'))), 8)
        worker.verify_progress(progress, self.inputs, self.binding)
        metrics = progress['report']['total']['r6_response']
        self.assertEqual((metrics['correct'], metrics['expected'], metrics['scored'], metrics['invalid']), (2, 4, 4, 1))
        self.assertEqual(progress['report']['total']['paired']['primary_delta_points'], 0)

    def test_completed_resume_does_not_load_model(self):
        self.run_engine()
        progress = worker.evaluate_local(self.root, self.inputs, self.binding,
                                         lambda: self.fail('Completed run must not reload'), attempt=1)
        self.assertEqual(progress['state'], 'complete')

    def test_paused_resume_only_scores_remaining(self):
        first, progress = self.run_engine(seconds=3, clock=Clock())
        self.assertEqual(progress['state'], 'paused')
        self.assertGreater(len(first.calls), 0)
        self.assertLess(len(first.calls), 8)
        second, progress = self.run_engine(attempt=1)
        self.assertEqual(len(first.calls) + len(second.calls), 8)
        self.assertFalse(set(first.calls) & set(second.calls))
        self.assertEqual(progress['state'], 'complete')

    def test_orphan_case_is_adopted_after_interrupted_checkpoint(self):
        _, progress = self.run_engine()
        progress['state'] = 'paused'
        progress['records'].pop()
        progress['report'] = p.report(self.inputs, progress['records'])
        progress['snapshot_sha256'] = p.digest({k: v for k, v in progress.items() if k != 'snapshot_sha256'})
        p.save(self.root / 'progress.json', progress)
        runtime, done = self.run_engine(attempt=1)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(len(done['records']), 8)

    def test_failed_case_remains_in_denominator_after_resume(self):
        with self.assertRaisesRegex(RuntimeError, 'Case failed'):
            self.run_engine(FakeRuntime(self.proofs, fail='mmlu:1'))
        _, progress = self.run_engine(attempt=1)
        metrics = progress['report']['total']['r6_response']
        self.assertEqual((metrics['expected'], metrics['errors'], metrics['scored']), (4, 1, 3))
        self.assertIsNone(metrics['final_accuracy'])
        self.assertIsNone(progress['report']['total']['paired']['primary_delta_points'])
        self.assertEqual(progress['report']['total']['paired']['scored_pairs'], 3)

    def test_mutated_or_missing_percase_artifact_fails(self):
        self.run_engine()
        path = next((self.root / 'cases').rglob('*.json'))
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'artifact missing'):
            self.run_engine(attempt=1)

    def test_corrupt_progress_score_rejected_even_if_resealed(self):
        _, progress = self.run_engine()
        progress['records'][0]['correct'] = False
        progress['snapshot_sha256'] = p.digest({k: v for k, v in progress.items() if k != 'snapshot_sha256'})
        with self.assertRaisesRegex(ValueError, 'Saved score differs'):
            worker.verify_progress(progress, self.inputs, self.binding)

    def test_paired_tokenization_difference_is_rejected(self):
        _, progress = self.run_engine()
        changed = deepcopy(progress['records'])
        changed[-1]['output']['prompt_ids_sha256'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'tokenized inputs differ'):
            p.report(self.inputs, changed)

    def test_duplicate_records_rejected(self):
        _, progress = self.run_engine()
        with self.assertRaisesRegex(ValueError, 'Duplicate reported case'):
            p.report(self.inputs, progress['records'] + progress['records'][:1])

    def test_loader_mismatch_fails_before_second_variant(self):
        self.proofs['r7_response']['common']['key_mapping'] = {}
        runtime = FakeRuntime(self.proofs)
        with self.assertRaisesRegex(ValueError, 'native loading proof'):
            self.run_engine(runtime)
        self.assertEqual({v for v, _ in runtime.calls}, {'r6_response'})

    def test_loading_failure_persists_no_successes(self):
        with self.assertRaises(RuntimeError):
            worker.evaluate_local(self.root, self.inputs, self.binding,
                                  lambda: (_ for _ in ()).throw(RuntimeError('loader unavailable')))
        progress = p.read(self.root / 'progress.json')
        worker.verify_progress(progress, self.inputs, self.binding)
        self.assertEqual(progress['records'], [])
        self.assertEqual(progress['state'], 'error')


def node(state, children=()):
    return SimpleNamespace(status=SimpleNamespace(name=state), children=list(children))


class FakeBackend:
    def __init__(self, fail_spawn=False):
        self.claims, self.spawned, self.fail_spawn = {}, 0, fail_spawn
        self.state = 'SUCCESS'
        self.checkpoint = None

    def claim(self, worker, run_id, attempt):
        return self.claims.get(attempt)

    def remember(self, worker, run_id, attempt, call_id, execution_sha):
        self.claims[attempt] = {'call_id': call_id, 'execution_sha256': execution_sha, 'attempt': attempt}

    def graph(self, call_id):
        return [node(self.state)]

    def progress(self, worker, run_id):
        return self.checkpoint

    def application(self, worker, package, execution_sha):
        def run(*, detach):
            assert detach is True
            return nullcontext()
        def spawn(attempt):
            self.spawned += 1
            if self.fail_spawn:
                raise ConnectionError('ambiguous submission')
            return SimpleNamespace(object_id='fc-test-' + str(self.spawned))
        return SimpleNamespace(app_id='ap-test', run=run), SimpleNamespace(spawn=spawn)

    def output(self):
        return nullcontext()


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        inputs, binding, _ = fixtures()
        binding['code'] = {'code/r7_public_benchmark_launch.py': p.sha(launch.__file__)}
        self.execution = {'binding': binding, 'bundle': str(self.root / 'bundle')}
        for target, value in (('worker_for', (worker, self.execution, inputs)),):
            mock = patch.object(launch, target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch.object(launch.core, 'bundle_module', return_value=({}, None))
        mock.start()
        self.addCleanup(mock.stop)
        mock = patch.object(worker, 'require_launch_python')
        mock.start()
        self.addCleanup(mock.stop)
        self.backend = FakeBackend()

    def call(self, **kwargs):
        return launch.launch(self.root / 'package', 'a' * 64, self.root / 'receipt.json',
                             backend=self.backend, **kwargs)

    def test_authorization_required_before_backend(self):
        with self.assertRaisesRegex(ValueError, 'authorize-gpu'):
            self.call()
        self.assertEqual(self.backend.spawned, 0)

    def test_detached_receipt_then_terminal_resume(self):
        first = self.call(authorize_gpu=True)
        self.assertEqual(first['state'], 'submitted')
        with self.assertRaisesRegex(ValueError, 'Existing receipt'):
            self.call(authorize_gpu=True)
        next_call = self.call(authorize_gpu=True, resume=True)
        self.assertEqual([a['attempt'] for a in next_call['attempts']], [0, 1])
        self.assertEqual(self.backend.spawned, 2)

    def test_active_call_cannot_be_resubmitted(self):
        self.call(authorize_gpu=True)
        self.backend.state = 'PENDING'
        with self.assertRaisesRegex(ValueError, 'still be active'):
            self.call(authorize_gpu=True, resume=True)
        self.assertEqual(self.backend.spawned, 1)

    def test_unknown_spawn_cannot_be_resubmitted(self):
        self.backend.fail_spawn = True
        with self.assertRaises(ConnectionError):
            self.call(authorize_gpu=True)
        self.assertEqual(p.read(self.root / 'receipt.json')['state'], 'submission_unknown')
        with self.assertRaisesRegex(ValueError, 'outcome unknown'):
            self.call(authorize_gpu=True, resume=True)
        self.assertEqual(self.backend.spawned, 1)

    def test_unknown_spawn_recovered_by_claim(self):
        self.backend.fail_spawn = True
        with self.assertRaises(ConnectionError):
            self.call(authorize_gpu=True)
        self.backend.remember(worker, 'unused', 0, 'fc-recovered', 'a' * 64)
        self.backend.fail_spawn = False
        result = self.call(authorize_gpu=True, resume=True)
        self.assertEqual(result['attempts'][0]['call_id'], 'fc-recovered')
        self.assertEqual(result['attempts'][1]['attempt'], 1)

    def test_terminal_requires_children_and_nonempty_graph(self):
        for graph in ([], [node('RUNNING')], [node('SUCCESS', [node('UNKNOWN')])]):
            with self.assertRaises(ValueError):
                worker.terminal_state(graph)
        self.assertEqual(worker.terminal_state([node('TIMEOUT')]), 'FAILED')


class RealArchiveTests(unittest.TestCase):
    def setUp(self):
        self.results = ROOT / 'tmp/r6-benchmark-v2-20261117/results'
        self.source_launch = self.results.parent / 'launch.json'
        if not self.results.exists():
            self.skipTest('Optional local corrected R6 archive not present')

    def test_exact_2004_inputs_and_protection_assembler_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'protection'
            result = protection.prepare(self.results, self.source_launch, out)
            manifest = protection.verify(out, result['manifest_sha256'])
            self.assertEqual(manifest['held_units'], 2004)
            from select_understanding_r5_sources import bound_package, bound_read, recheck
            pins = {}
            bound_package(out, pins, {'held-content.jsonl'}, 'SEALED.json')
            rows = bound_read(out / 'held-content.jsonl', pins, rows=True)
            recheck(pins)
            self.assertEqual(len(rows), 2004)
            self.assertTrue(all(r['text'] == r['prompt'] and r['training_eligible'] is False for r in rows))
            self.assertFalse(manifest['foundation_unseen_claim'])
            with self.assertRaisesRegex(ValueError, 'new protection'):
                protection.prepare(self.results, self.source_launch, out)

    def test_mutated_input_cannot_reseal_as_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for name in ('manifest.json', 'inputs.json', 'SEALED.json'):
                shutil.copyfile(self.results / name, out / name)
            inputs = p.read(out / 'inputs.json')
            inputs['mmlu'][0]['prompt'] += ' altered'
            p.save(out / 'inputs.json', inputs)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                protection.source_inputs(out, self.source_launch)

    def test_historical_audit_recomputes_all_6012_cases(self):
        audit = p.verify_historical(self.results, self.source_launch)
        self.assertFalse(audit['eligible_as_verified_r6_comparator'])
        self.assertEqual(audit['cases_per_variant'], 2004)
        self.assertEqual(sum(v['total'] for v in audit['historical_scores_only']['r6'].values()), 2004)

    def test_frozen_archive_verifies(self):
        frozen = ROOT / 'tmp/r7-20261117/public-benchmark-frozen-v1'
        if not frozen.exists():
            self.skipTest('Optional local frozen copy absent')
        manifest, inputs = p.verify_frozen(frozen, p.sha(frozen / 'manifest.json'))
        self.assertEqual(manifest['cases'], sum(map(len, inputs.values())))

    def test_r6_only_bundle_rejected_before_gpu(self):
        bundle = ROOT / 'tmp/r6-demo/bundle-v1'
        if not bundle.exists():
            self.skipTest('Optional R6 bundle absent')
        with self.assertRaisesRegex(ValueError, 'Completed R7'):
            p.bundle_module(bundle, p.R6_BUNDLE_SHA)


class ImportTests(unittest.TestCase):
    def test_import_does_not_import_modal_or_torch(self):
        code = "import sys; sys.path.insert(0, 'scripts'); import r7_public_benchmark_protocol as p; " \
               "p.import_file('modal/train/r7_public_benchmark.py'); " \
               "import r7_public_benchmark_launch; assert 'modal' not in sys.modules; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ProtectionAuditTests(unittest.TestCase):
    def test_missing_field_is_scanned_independently_and_embedded_hash_detected(self):
        held = [{'source_group_id': 'public-group', 'source_id': 'public-source',
                 'original_ref': {'source_sha256': 'a' * 64}}]
        fields = {'assembler': {'group_id', 'group_aliases', 'source_id'}}
        report, seen = protection_audit.scan([{'id': 'clean', 'source_id': 'other'}], held, fields)
        self.assertTrue(report['zero_intersection'])
        self.assertEqual(seen, {'clean'})
        contaminated = {'id': 'hit', 'metadata': {'source_group_id': 'public-group', 'reference': 'sha256:' + 'a' * 64}}
        report, _ = protection_audit.scan([contaminated], held, fields)
        self.assertFalse(report['zero_intersection'])
        matches = report['matching_witnesses'][0]['matches']
        self.assertEqual(matches['all_string_fields:source_group_id'], ['public-group'])
        self.assertEqual(matches['embedded_source_sha256'], ['a' * 64])
        self.assertEqual(report['intersection_counts']['assembler']['source_group_id'], 0)

    def test_recognized_group_alias_intersection_detected(self):
        held = [{'source_group_id': 'group', 'source_id': 'source', 'original_ref': {'source_sha256': 'b' * 64}}]
        report, _ = protection_audit.scan([{'id': 'x', 'metadata': {'group_aliases': ['group']}}], held,
                                          {'runner': {'group_aliases'}})
        self.assertEqual(report['intersection_counts']['runner']['source_group_id'], 1)


if __name__ == '__main__':
    unittest.main()
