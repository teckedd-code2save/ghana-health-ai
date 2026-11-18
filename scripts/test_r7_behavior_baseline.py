"""Offline worker tests; all model results and invocation claims are synthetic."""
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

import run_r7_behavior_baseline as launcher

worker = launcher.load_worker()


def plan():
    return {'version': worker.PLAN_VERSION,
            'cases': [{'id': 'fixed-a', 'messages': [{'role': 'user', 'content': 'fixed-a'}]},
                      {'id': 'fixed-b', 'messages': [{'role': 'user', 'content': 'fixed-b'}]}],
            'rollouts': [{'id': 'roll-a', 'system_message': 'system-a', 'user_turns': ['a0', 'a1', 'a2']},
                         {'id': 'roll-b', 'system_message': 'system-b', 'user_turns': ['b0', 'b1', 'b2']}]}


def binding(value):
    return {'plan_sha256': worker.validate(value), 'guard': 'raw', 'run_id': 'r7-baseline-synthetic'}


def node(status, children=()):
    return SimpleNamespace(status=SimpleNamespace(name=status), children=list(children))


class Runtime:
    def __init__(self, actions=None):
        self.actions = actions or {}
        self.calls = []

    def reply(self, messages, on_text):
        self.calls.append(deepcopy(messages))
        question = messages[-1]['content']
        action = self.actions.get(question)
        if isinstance(action, BaseException):
            on_text('partial before error')
            raise action
        return action or {'output': 'reply:' + question, 'complete': True}


class ClaimStore(dict):
    def put(self, key, value, skip_if_exists=False):
        if not skip_if_exists or key not in self:
            self[key] = value


class EngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = plan()
        self.binding = binding(self.plan)

    def run_engine(self, runtime=None, **kwargs):
        runtime = runtime or Runtime()
        return worker.evaluate_local(self.root, self.plan, self.binding, lambda: runtime, **kwargs)

    def test_system_messages_independent_histories_and_all_denominators(self):
        runtime = Runtime()
        report = self.run_engine(runtime)
        self.assertEqual(report['counts']['expected_slots'], 8)
        self.assertEqual(report['counts']['states']['complete'], 8)
        self.assertEqual(len(runtime.calls), 8)
        a1 = next(messages for messages in runtime.calls if messages[-1]['content'] == 'a1')
        self.assertEqual(a1, [{'role': 'system', 'content': 'system-a'}, {'role': 'user', 'content': 'a0'},
                              {'role': 'assistant', 'content': 'reply:a0'}, {'role': 'user', 'content': 'a1'}])
        b0 = next(messages for messages in runtime.calls if messages[-1]['content'] == 'b0')
        self.assertEqual([m['content'] for m in b0], ['system-b', 'b0'])
        worker.verify_report(worker.read(self.root / 'progress.json'), self.plan, self.binding)

    def test_empty_error_and_incomplete_block_only_their_rollout(self):
        for action, state in (({'output': '', 'complete': False}, 'invalid'),
                              (RuntimeError('synthetic error'), 'error'),
                              ({'output': 'partial', 'complete': False}, 'incomplete')):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as folder:
                runtime = Runtime({'a0': action})
                report = worker.evaluate_local(folder, self.plan, self.binding, lambda: runtime)
                records = {r['id']: r for r in report['results']}
                self.assertEqual(records['roll-a:turn-0']['state'], state)
                self.assertEqual(records['roll-a:turn-1']['state'], 'blocked')
                self.assertEqual(records['roll-a:turn-2']['blocked_by'], 'roll-a:turn-0')
                self.assertEqual(records['roll-b:turn-2']['state'], 'complete')
                self.assertEqual(len(runtime.calls), 6)
                self.assertEqual(sum(report['counts']['states'].values()), 8)
                self.assertEqual(report['counts']['pending_slots'], 0)
                worker.verify_report(report, self.plan, self.binding)

    def test_fixed_error_keeps_partial_text_and_continues_other_cases(self):
        report = self.run_engine(Runtime({'fixed-a': RuntimeError('fixture')}))
        self.assertEqual(report['results'][0]['output'], 'partial before error')
        self.assertEqual(report['results'][0]['state'], 'error')
        self.assertEqual(report['results'][1]['state'], 'complete')

    def test_runtime_loading_failure_is_checkpointed_with_all_slots_pending(self):
        def fail():
            raise RuntimeError('synthetic load failure')
        report = worker.evaluate_local(self.root, self.plan, self.binding, fail)
        self.assertEqual(report['state'], 'error')
        self.assertEqual(report['error']['phase'], 'runtime_loading')
        self.assertEqual(report['counts']['pending_slots'], 8)
        worker.verify_report(worker.read(self.root / 'progress.json'), self.plan, self.binding)
        self.assertEqual(self.run_engine(attempt=1)['counts']['finalized_slots'], 8)

    def test_finished_inputs_never_reload_or_rerun(self):
        report = self.run_engine()
        def forbidden():
            self.fail('Completed report loaded a runtime')
        again = worker.evaluate_local(self.root, self.plan, self.binding, forbidden, attempt=1)
        self.assertEqual(again, report)

    def test_partial_resume_does_not_repeat_saved_fixed_input(self):
        elapsed = [0]
        runtime = Runtime()
        original = runtime.reply
        def reply(messages, on_text):
            result = original(messages, on_text)
            elapsed[0] += 2
            return result
        runtime.reply = reply
        report = self.run_engine(runtime, clock=lambda: elapsed[0], seconds=1)
        self.assertEqual(report['state'], 'paused')
        self.assertEqual(report['counts']['finalized_slots'], 1)
        resumed = Runtime()
        report = self.run_engine(resumed, attempt=1)
        self.assertEqual(len(resumed.calls), 7)
        self.assertNotIn('fixed-a', [messages[-1]['content'] for messages in resumed.calls])
        self.assertEqual(report['counts']['finalized_slots'], 8)

    def test_crash_after_request_checkpoint_is_not_silently_repeated(self):
        class SimulatedTermination(BaseException):
            pass
        runtime = Runtime({'fixed-b': SimulatedTermination()})
        with self.assertRaises(SimulatedTermination):
            self.run_engine(runtime)
        checkpoint = worker.read(self.root / 'progress.json')
        self.assertEqual(checkpoint['inflight']['id'], 'fixed-b')
        resumed = Runtime()
        report = self.run_engine(resumed, attempt=1)
        self.assertEqual(len(resumed.calls), 6)
        self.assertEqual(report['results'][1]['error']['type'], 'InterruptedWithoutResult')
        worker.verify_report(report, self.plan, self.binding)

    def test_native_validation_error_preserves_rejected_raw_result(self):
        raw = {'output': 'untrusted result', 'complete': True, 'identity': 'wrong'}
        report = self.run_engine(Runtime({'fixed-a': worker.InvalidNativeResult('identity changed', raw)}))
        self.assertEqual(report['results'][0]['result'], raw)
        self.assertFalse(report['results'][0]['verified'])
        self.assertEqual(report['results'][0]['state'], 'error')

    def test_checkpoint_error_does_not_duplicate_a_result(self):
        failed = [False]
        def commit():
            value = worker.read(self.root / 'progress.json')
            if value['counts']['finalized_slots'] == 1 and not failed[0]:
                failed[0] = True
                raise RuntimeError('one simulated commit error')
        with self.assertRaisesRegex(RuntimeError, 'commit error'):
            self.run_engine(commit=commit)
        self.assertEqual(len(worker.read(self.root / 'progress.json')['results']), 1)
        runtime = Runtime()
        report = self.run_engine(runtime, attempt=1)
        self.assertEqual(len(runtime.calls), 7)
        self.assertEqual(len({r['id'] for r in report['results']}), 8)

    def test_altered_snapshot_or_resealed_history_is_rejected_before_runtime(self):
        report = deepcopy(self.run_engine())
        report['results'][0]['output'] = 'changed'
        worker.save(self.root / 'progress.json', report)
        with self.assertRaisesRegex(ValueError, 'snapshot hash'):
            self.run_engine()
        report['results'][0]['messages'][0]['content'] = 'another history'
        report['snapshot_sha256'] = worker.digest({k: v for k, v in report.items() if k != 'snapshot_sha256'})
        worker.save(self.root / 'progress.json', report)
        with self.assertRaisesRegex(ValueError, 'history/slot'):
            self.run_engine()


class ContractTests(unittest.TestCase):
    def test_gold_fields_system_omission_duplicate_ids_and_roles_rejected(self):
        for mutation in ('gold', 'system', 'duplicate', 'role', 'collision'):
            value = plan()
            if mutation == 'gold':
                value['cases'][0]['constraints'] = []
            elif mutation == 'system':
                del value['rollouts'][0]['system_message']
            elif mutation == 'duplicate':
                value['cases'][1]['id'] = 'fixed-a'
            elif mutation == 'role':
                value['cases'][0]['messages'][0]['role'] = 'assistant'
            else:
                value['cases'][0]['id'] = 'roll-a:turn-0'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker.validate(value)

    def test_only_explicit_terminal_graphs_allow_resume(self):
        self.assertEqual(worker.terminal_state([node('SUCCESS')]), 'SUCCESS')
        self.assertEqual(worker.terminal_state([node('TIMEOUT')]), 'FAILED')
        for graph in ([], [node('RUNNING')], [node('SUCCESS', [node('PENDING')])]):
            with self.assertRaises(ValueError):
                worker.terminal_state(graph)

    def test_attempt_claims_are_atomic_and_never_take_over_running_work(self):
        claims, identity = ClaimStore(), {'fixture': True}
        worker.claim_attempt(claims, 'r7-baseline-synthetic', 0, 'fc-one', identity, lambda _: [])
        with self.assertRaisesRegex(ValueError, 'already owned'):
            worker.claim_attempt(claims, 'r7-baseline-synthetic', 0, 'fc-two', identity, lambda _: [])
        with self.assertRaisesRegex(ValueError, 'still be active'):
            worker.claim_attempt(claims, 'r7-baseline-synthetic', 1, 'fc-three', identity, lambda _: [node('RUNNING')])
        worker.claim_attempt(claims, 'r7-baseline-synthetic', 1, 'fc-three', identity, lambda _: [node('FAILURE')])
        self.assertEqual(len(claims), 2)

    def test_unverified_bundle_code_is_never_executed(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder)
            files = ('r6_inference.py', 'r2_inference.py', 'afrique_instruction_baseline_core.py', 'understanding_r2_demo.py')
            (bundle / 'code').mkdir()
            for name in files:
                (bundle / 'code' / name).write_text('raise AssertionError("MUST NOT EXECUTE")\n')
            manifest = {'version': 'r6-comparison-inference-v1', 'training': False, 'private': True,
                        'files': {'code/' + name: worker.sha(bundle / 'code' / name) for name in files}}
            worker.save(bundle / 'manifest.json', manifest)
            checksum = worker.sha(bundle / 'manifest.json')
            worker.save(bundle / 'SEALED.json', {'manifest_sha256': checksum})
            (bundle / 'code/r2_inference.py').write_text('changed dependency\n')
            with self.assertRaisesRegex(ValueError, 'Bundle artifact changed'):
                worker.load_runtime(bundle, checksum)

    def test_real_frozen_bundle_hashes_verify_without_runtime_import(self):
        bundle = launcher.ROOT / 'tmp/r6-demo/bundle-v1'
        manifest = worker.verify_bundle(bundle, worker.EXPECTED)
        self.assertEqual(manifest['default_variant'], 'r6_response')
        self.assertFalse(manifest['training'])

    def test_actual_modal_serialization_does_not_capture_lazy_claims(self):
        import modal
        from modal._serialization import serialize
        bundle = launcher.ROOT / 'tmp/r6-demo/bundle-v1'
        # Version compatibility is tested separately; this test isolates lazy handles.
        with patch.object(worker, 'require_launch_python'):
            app, function, claims = worker.create_app(bundle, worker.EXPECTED)
        self.assertEqual(app.name, worker.APP_NAME)
        self.assertFalse(claims.is_hydrated)
        raw = inspect.unwrap(function.get_raw_f())
        self.assertNotIn('claims', raw.__code__.co_freevars)
        # App startup hydrates mounted volumes. Substitute only that mounted handle
        # so the real SDK serializer can test every other closure value offline.
        def cell(value):
            return (lambda: value).__closure__[0]
        closure = tuple(cell('mounted-volume-placeholder') if name == 'volume' else value
                        for name, value in zip(raw.__code__.co_freevars, raw.__closure__))
        probe = FunctionType(raw.__code__, raw.__globals__, raw.__name__, raw.__defaults__, closure)
        self.assertGreater(len(serialize(probe)), 100)
        with self.assertRaisesRegex(modal.exception.InvalidError, 'hydrated'):
            serialize(claims)

    def test_launcher_can_create_claim_dictionary_before_container_starts(self):
        store = ClaimStore()
        calls = []
        def from_name(name, **kwargs):
            calls.append((name, kwargs))
            return store
        backend = object.__new__(launcher.ModalBackend)
        backend.modal = SimpleNamespace(Dict=SimpleNamespace(from_name=from_name))
        backend.remember(worker, 'r7-baseline-synthetic', 0, 'fc-synthetic', {'fixture': True})
        self.assertEqual(calls, [(worker.CLAIMS, {'create_if_missing': True})])

    def test_serialized_python_version_preflight_rejects_before_resource_setup(self):
        for version, implementation in (((3, 13, 0), 'cpython'), ((3, 14, 0), 'cpython'),
                                         ((3, 11, 15), 'pypy')):
            runtime = SimpleNamespace(version_info=version, implementation=SimpleNamespace(name=implementation))
            with self.subTest(version=version, implementation=implementation), patch.object(worker, 'sys', runtime), \
                    patch.object(worker, 'verify_bundle') as verify:
                with self.assertRaisesRegex(ValueError, 'requires CPython 3.11'):
                    worker.create_app('/unused', worker.EXPECTED, modal_module=object())
                verify.assert_not_called()
        runtime = SimpleNamespace(version_info=(3, 11, 15), implementation=SimpleNamespace(name='cpython'),
                                  version='3.11.15 fixture', executable='/fixture/python3.11')
        with patch.object(worker, 'sys', runtime):
            self.assertEqual(worker.require_launch_python(), {'implementation': 'cpython', 'version': '3.11.15',
                             'executable': '/fixture/python3.11', 'image_python': '3.11'})

    def test_launcher_rejects_wrong_python_before_receipt_or_modal_backend(self):
        execution = {'launcher_sha256': launcher.file_sha(launcher.__file__)}
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(launcher, 'verify_package', return_value=(worker, execution, {}, [])), \
                patch.object(worker, 'require_launch_python', side_effect=ValueError('requires CPython 3.11')), \
                patch.object(launcher, 'ModalBackend') as backend:
            receipt = Path(folder) / 'receipt.json'
            with self.assertRaisesRegex(ValueError, 'requires CPython 3.11'):
                launcher.launch('/unused', 'fixture', receipt, authorize_gpu=True)
            backend.assert_not_called()
            self.assertFalse(receipt.exists())


if __name__ == '__main__':
    unittest.main()
