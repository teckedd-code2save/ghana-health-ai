import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import r6_benchmark_protocol as p
from collect_r6_benchmark import verify, paired


class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row = p.normalize('mmlu', {'question':'Question', 'choices':['one','two'], 'answer':0}, 0)
        inputs = {'mmlu':[self.row]}
        legacy = {}
        for variant in p.VARIANTS:
            path = self.root / 'legacy' / f'{variant}.json'
            self.save(path, {'variant':variant})
            legacy[variant] = {'sha256': self.sha(path)}
        manifest = {'run_id':'test-run', 'model':'pinned', 'revision':'revision', 'adapters':{'r5':{},'r6':{}},
                    'code':{}, 'inputs_sha256':p.digest(inputs), 'legacy':legacy}
        self.save(self.root / 'inputs.json', inputs)
        self.save(self.root / 'manifest.json', manifest)
        self.manifest_sha = self.sha(self.root / 'manifest.json')
        self.save(self.root / 'SEALED.json', {'manifest_sha256':self.manifest_sha})
        for variant in p.VARIANTS:
            output = {**self.row, 'scores':{'A':-1.,'B':-2.}, 'prediction':'A', 'correct':True,
                      'manifest_sha256':self.manifest_sha, 'variant':variant}
            self.save(self.root / f'predictions/{variant}/mmlu/000000.json', output)
            summary = {'state':'completed', 'manifest_sha256':self.manifest_sha, 'model':'pinned',
                       'revision':'revision', 'runtime':{}, 'tokenizer_sha256':'tok', 'chat_template_sha256':'template',
                       'adapter':manifest['adapters'].get(variant), 'benchmarks':{'mmlu':p.summarize([output])}}
            self.save(self.root / f'{variant}.json', summary)

    @staticmethod
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def mutate(self, relative, update):
        path = self.root / relative
        value = json.loads(path.read_text())
        update(value)
        self.save(path, value)

    def test_complete_reproducible_scores(self):
        report = verify(self.root)
        self.assertEqual(report['tasks']['mmlu']['r6']['correct'], 1)
        self.assertEqual(report['tasks']['mmlu']['r6_vs_r5']['ties'], 1)

    def test_wrong_adapter_cannot_pass(self):
        self.mutate('r6.json', lambda x: x.update(adapter={'wrong':True}))
        with self.assertRaisesRegex(ValueError, 'Checkpoint'):
            verify(self.root)

    def test_wrong_prediction_cannot_pass(self):
        self.mutate('predictions/r6/mmlu/000000.json', lambda x: x.update(prediction='B', correct=False))
        with self.assertRaisesRegex(ValueError, 'Score mismatch'):
            verify(self.root)

    def test_changed_prompt_cannot_pass(self):
        self.mutate('predictions/r6/mmlu/000000.json', lambda x: x.update(prompt='Other input'))
        with self.assertRaisesRegex(ValueError, 'Row identity'):
            verify(self.root)

    def test_missing_row_cannot_pass(self):
        (self.root / 'predictions/r6/mmlu/000000.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Wrong result count'):
            verify(self.root)

    def test_changed_historical_results_cannot_pass(self):
        self.save(self.root / 'legacy/base.json', {'changed':True})
        with self.assertRaisesRegex(ValueError, 'Legacy changed'):
            verify(self.root)

    def test_float_summation_rounding_only_in_chance_is_allowed(self):
        self.mutate('r6.json', lambda x: x['benchmarks']['mmlu'].update(chance=0.5-1e-15))
        self.assertTrue(verify(self.root)['verified'])

    def test_changed_accuracy_cannot_pass_as_rounding(self):
        self.mutate('r6.json', lambda x: x['benchmarks']['mmlu'].update(accuracy=1-1e-15))
        with self.assertRaisesRegex(ValueError, 'Aggregate mismatch'):
            verify(self.root)

    def test_paired_gains_and_losses(self):
        before = [{'correct':v} for v in [False, False, True, True]]
        after = [{'correct':v} for v in [True, True, False, True]]
        report = paired(before, after)
        self.assertEqual((report['gains'],report['losses'],report['ties']), (2,1,1))
        self.assertEqual(report['delta_points'], 25)


if __name__ == '__main__':
    unittest.main()
