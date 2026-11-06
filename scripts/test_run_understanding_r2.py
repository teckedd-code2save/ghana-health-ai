"""CPU orchestration tests; no Modal GPU submission or model download."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_understanding_r2 as r

SOURCE = Path(os.environ.get('GHA_R2_SOURCE_ROOT', '/Users/welcome/Documents/SoftwareEngineering/serendepify/ghana-health-ai'))


class ContractTests(unittest.TestCase):
    def test_actual_frozen_helpers_match_successful_r1(self):
        r.activate(SOURCE)

    def test_mount_alias_is_resolved_but_run_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            physical = root / 'volume'
            physical.mkdir()
            alias = root / 'artifacts'
            alias.symlink_to(physical, target_is_directory=True)
            self.assertEqual(r.mounted_root(alias), physical / r.VERSION)
            (physical / r.VERSION).rmdir()
            (physical / r.VERSION).symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                r.mounted_root(alias)

    def test_real_worker_bounds_and_no_r1_write_root(self):
        source = (r.ROOT / 'modal/train/train_understanding_r2.py').read_text()
        tree = ast.parse(source)
        for name, timeout in (('smoke', 900), ('train', 14400)):
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            options = {k.arg: k.value for k in fn.decorator_list[0].keywords}
            for key, expected in {'timeout': timeout, 'min_containers': 0, 'max_containers': 1,
                                  'retries': 0, 'gpu': 'H100', 'cpu': 4, 'memory': 49152}.items():
                self.assertEqual(ast.literal_eval(options[key]), expected)
        self.assertNotIn('/understanding-adaptation-r1', source)
        self.assertIn('torch.manual_seed(42)', source)
        self.assertNotIn('torch.manual_seed(40)', source)
        self.assertIn('max_seconds=max(0, remaining - 60)', source)
        self.assertIn("'PYTHONPATH': '/r2-package/r2/modal/train:", source)

    def test_terminal_pause_rejects_pending_completed_or_wrong_identity(self):
        claim = {'call_id': 'fc-test', 'execution_sha256': 'a' * 64}
        value = {'state': 'paused_budget', 'mode': 'train', 'training_completed': False,
                 'run_id': 'understanding_20261105T120000Z', 'execution_sha256': 'a' * 64, 'data_sha256': 'b' * 64}
        self.assertEqual(r.terminal_pause(claim, value, value['run_id'], 'a' * 64, 'b' * 64)['state'], 'completed')
        for change in ({'state': 'training'}, {'state': 'completed'}, {'run_id': '../other'}, {'data_sha256': 'c' * 64}):
            with self.assertRaises(ValueError):
                r.terminal_pause(claim, {**value, **change}, value['run_id'], 'a' * 64, 'b' * 64)

    def test_launch_requires_authorization_and_records_before_import(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            root = Path(temporary)
            receipt = root / 'call.json'
            r.save(root / 'execution.json', {'test': True})
            contract = {'mode': 'smoke', 'data_sha256': 'b' * 64}
            with self.assertRaises(ValueError):
                r.launch(root, receipt, 'test', False)
            def fail_import(name, path):
                self.assertEqual(r.read(receipt)['state'], 'submitting')
                raise RuntimeError('simulated pre-submission crash')
            with patch.object(r, 'verify_execution', return_value=contract), patch.object(importlib.util, 'spec_from_file_location', side_effect=fail_import):
                with self.assertRaisesRegex(RuntimeError, 'pre-submission'):
                    r.launch(root, receipt, 'test', True)
                with self.assertRaisesRegex(ValueError, 'Submission exists'):
                    r.launch(root, receipt, 'test', True)
            self.assertNotIn('call_id', r.read(receipt))

    def test_empty_evaluation_is_not_a_verified_pack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r.save(root / 'manifest.json', {})
            with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                r.evaluation_pack(root, root)

    def test_smoke_requires_matching_evaluation_code_and_bitwise_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = {'mode': 'smoke', 'version': r.VERSION, 'data_sha256': 'b' * 64, 'runtime': r.RUNTIME,
                'frozen_code_hashes': r.FROZEN, 'code_hashes': {}, 'evaluation': {'sha256': 'c' * 64}, 'expected_stages': []}
            r.save(root / 'execution.json', old)
            r.save(root / 'status.json', {'state': 'completed', 'mode': 'smoke', 'run_id': 'understanding_20261105T120000Z',
                'execution_sha256': r.sha(root / 'execution.json'), 'data_sha256': 'b' * 64,
                'remote_directory': '/artifacts/' + r.VERSION + '/understanding_20261105T120000Z'})
            smoke = {'passed': True, 'real_weight_backward': True, 'full_state_recovery': {'equivalent': True},
                'resume_parameter_and_buffer_bitwise_equal': True, 'native_encoding_kinds': ['language', 'chat']}
            r.save(root / 'smoke.json', smoke)
            r.compare_smoke(root, {**old, 'mode': 'train'})
            with self.assertRaisesRegex(ValueError, 'differs'):
                r.compare_smoke(root, {**old, 'evaluation': {'sha256': 'd' * 64}})
            r.save(root / 'smoke.json', {**smoke, 'full_state_recovery': {'equivalent': False}})
            with self.assertRaises(ValueError):
                r.compare_smoke(root, old)


class RecoveryTests(unittest.TestCase):
    def test_full_state_copy_keeps_prior_adapter_and_recovers_next_stage(self):
        import torch
        from torch import nn
        r.activate(SOURCE)
        from understanding_training_core import train_schedule, assert_checkpoint_equivalent
        from understanding_recovery import select_recovery

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(8, 4)
                self.embedding.requires_grad_(False)
                self.lora = nn.Linear(4, 8, bias=False)
            def forward(self, input_ids, **kwargs):
                return {'logits': self.lora(self.embedding(input_ids))}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            previous = root / 'understanding_20261105T120000Z'
            destination = root / 'understanding_20261105T130000Z'
            previous.mkdir(); destination.mkdir()
            execution = {'mode': 'train', 'data_sha256': 'b' * 64}
            r.save(previous / 'inputs/manifest.json', {'data': True})
            execution['data_sha256'] = r.sha(previous / 'inputs/manifest.json')
            r.save(previous / 'execution.json', execution)
            execution_sha = r.sha(previous / 'execution.json')
            data_sha = execution['data_sha256']
            rows = [{'input_ids': [1, 2, 3], 'labels': [1, 2, 3], 'attention_mask': [1, 1, 1]}] * 2
            stages = [{'name': name, 'schedule': [0, 1], 'optimizer_updates': 2,
                'effective_target_tokens': 4, 'sequence_tokens': 6} for name in ('language', 'meaning', 'responses')]
            torch.manual_seed(42)
            model = Toy()
            options = {'run_sha256': execution_sha, 'data_sha256': data_sha, 'grad_accum': 1, 'checkpoint_every': 1}
            first = train_schedule(model, rows, [0, 1], output=previous / 'language', **options)
            second = train_schedule(model, rows, [0, 1], output=previous / 'meaning', stop_after_updates=1, **options)
            records = [{'stage': 'language', **first}, {'stage': 'meaning', **second}]
            r.save(previous / 'stages.json', records)
            r.save(previous / 'status.json', {'state': 'paused_budget', 'mode': 'train', 'run_id': previous.name,
                'execution_sha256': execution_sha, 'data_sha256': data_sha, 'stages': records})
            adapter = previous / 'language-adapter'
            adapter.mkdir(); (adapter / 'adapter_model.safetensors').write_bytes(b'test-only-adapter')
            files = {'adapter_model.safetensors': r.sha(adapter / 'adapter_model.safetensors')}
            r.save(previous / 'adapters.json', [{'stage': 'language', 'files': files}])
            claim = {'call_id': 'fc-test', 'execution_sha256': execution_sha}
            selected = select_recovery(previous, runs_root=root, execution_sha256=execution_sha,
                data_sha256=data_sha, stages=stages, grad_accum=1, claim=claim,
                terminal_call={'call_id': 'fc-test', 'state': 'completed'})
            def publish(path, checkpoint, run, data):
                self.assertEqual(r.sha(path / 'adapter_model.safetensors'), files['adapter_model.safetensors'])
                return {'stage': 'language', 'files': files, 'checkpoint': r.checkpoint_proof(checkpoint, run, data)}
            with patch.object(r, 'publish_adapter', side_effect=publish):
                inherited, adapters, selected = r.inherit_prefix(previous, destination, stages, selected, execution)
            self.assertEqual(len(inherited), 1)
            self.assertEqual(len(adapters), 1)
            self.assertTrue((destination / 'language-adapter/adapter_model.safetensors').is_file())
            self.assertTrue(assert_checkpoint_equivalent(second['checkpoint'], selected['checkpoint'])['equivalent'])
            torch.manual_seed(42)
            recovered = Toy()
            finished = train_schedule(recovered, rows, [0, 1], output=destination / 'meaning',
                resume=Path(selected['checkpoint']), **options)
            expected = train_schedule(model, rows, [0, 1], output=previous / 'meaning',
                resume=Path(second['checkpoint']), **options)
            self.assertTrue(assert_checkpoint_equivalent(finished['checkpoint'], expected['checkpoint'])['equivalent'])
            self.assertEqual(finished['position']['completed_microbatches'], 2)
            (Path(selected['checkpoint']) / 'state.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                r.checkpoint_proof(selected['checkpoint'], execution_sha, data_sha)


if __name__ == '__main__':
    unittest.main()
