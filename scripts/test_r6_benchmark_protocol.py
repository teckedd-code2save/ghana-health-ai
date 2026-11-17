import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import r6_benchmark_protocol as p


class ProtocolTest(unittest.TestCase):
    def test_serialized_choices_are_not_characters(self):
        row = p.normalize('afrimmlu_twi', {'question': 'Q?', 'choices': "['p = 4', 'p = 8', 'p = 12', 'p = 24']", 'answer': 'C'}, 0)
        self.assertIn('C. p = 12', row['prompt'])
        self.assertEqual(row['labels'], list('ABCD'))
        self.assertEqual(row['gold'], 'C')

    def test_variable_option_count(self):
        row = p.mcq_row('Q?', ['yes', 'no', 'maybe'], 'B')
        self.assertEqual(row['labels'], list('ABC'))
        with self.assertRaises(ValueError):
            p.choose(dict(A=1, B=2, C=3, D=999), row['labels'])
        self.assertEqual(p.choose(dict(A=1, B=2, C=3), row['labels']), 'C')
        self.assertIn('E. fifth', p.mcq_row('Q?', ['one','two','three','four','fifth'], 'E')['prompt'])

    def test_numeric_arc_labels_remap_by_source_position(self):
        row = p.normalize('arc_challenge', {'question':'Q?', 'choices':{'text':['x','y','z'], 'label':['1','2','3']}, 'answerKey':'2'}, 0)
        self.assertEqual(row['gold'], 'B')

    def test_invalid_source_fails_without_skipping(self):
        for value in ('not a list', "['x']", "['x', '']", "{'A': 'x'}"):
            with self.assertRaises((ValueError, SyntaxError)):
                p.choices(value)
        with self.assertRaises(ValueError):
            p.mcq_row('Q?', ['x','y'], 'C')
        with self.assertRaises(ValueError):
            p.normalize('medqa', {'question':'Q?', 'choices':['x','y'], 'answer':['z']}, 0)

    def test_classification_is_not_gold_substring(self):
        for value in ('politics sports health', '', 'health or travel', 'unhealthy', 'The answer is health because...'):
            self.assertIsNone(p.classify_output(value))
        self.assertEqual(p.classify_output('**Health**.'), 'health')
        self.assertEqual(p.classify_output('Category: science/technology'), 'science/technology')

    def test_token_continuations_not_first_token_only(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {'prompt':[1,2], 'promptA':[1,2,3], 'promptB':[1,2,4,5]}[text]
        ids, choices = p.answer_tokens(Tokenizer(), 'prompt', ['A','B'])
        self.assertEqual(ids, [1,2])
        self.assertEqual(choices['B'], [4,5])

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            p.choose({'A': float('nan'), 'B': 1.0}, ['A','B'])

    def test_source_hash_changes_on_source_edit(self):
        source = {'question':'Q?', 'choices':['x','y'], 'answer':0}
        before = p.normalize('mmlu', source, 0)
        source['question'] = 'Other?'
        after = p.normalize('mmlu', source, 0)
        self.assertNotEqual(before['source_sha256'], after['source_sha256'])

    def test_boundary_merge_and_duplicate_encodings_rejected(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [1] if text == 'P' else [2,3]
        with self.assertRaises(ValueError):
            p.answer_tokens(Tokenizer(), 'P', ['A','B'])

    def test_duplicate_evidence_rejected(self):
        row = {'id':'x', 'correct':True, 'prediction':'A', 'labels':['A','B']}
        with self.assertRaises(ValueError):
            p.summarize([row, row])

    def test_resume_requires_terminal_evidence(self):
        from benchmark_launch import terminal_state
        node = lambda status: SimpleNamespace(status=SimpleNamespace(name=status), children=[])
        for graph in ([], [node('PENDING')], [node('UNKNOWN')]):
            with self.assertRaises(ValueError):
                terminal_state(graph)
        self.assertEqual(terminal_state([node('SUCCESS')]), 'SUCCESS')
        self.assertEqual(terminal_state([node('FAILURE')]), 'FAILED')


class ScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        path = Path(__file__).resolve().parents[1] / 'modal/train/benchmark_r6.py'
        spec = importlib.util.spec_from_file_location('_test_benchmark_r6', path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_only_presented_choices_get_scored(self):
        torch = self.torch
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                assert kwargs['enable_thinking'] is False
                return 'P'
            def encode(self, text, **kwargs):
                return [0] + ([ord(text[-1]) - 64] if len(text) > 1 else [])
        class Model:
            device = 'cpu'
            def __call__(self, **kwargs):
                # E would win under the broken scorer; it is not an offered answer.
                return SimpleNamespace(logits=torch.tensor([[[0., 1., 3., 2., 4., 99.]]]))
        result = self.runner.score_mcq(Model(), Tokenizer(), p.mcq_row('Q?', ['x','y','z'], 'B'))
        self.assertEqual(result['prediction'], 'B')
        self.assertEqual(set(result['scores']), set('ABC'))

    def test_multitoken_candidate_scores_all_tokens(self):
        torch = self.torch
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return 'P'
            def encode(self, text, **kwargs):
                return {'P':[0], 'PA':[0,1], 'PB':[0,2,3]}[text]
        class Model:
            device = 'cpu'
            def __call__(self, **kwargs):
                if kwargs['logits_to_keep'] == 1:
                    return SimpleNamespace(logits=torch.tensor([[[0., 2., 3., 0.]]]))
                return SimpleNamespace(logits=torch.tensor([[[0., 2., 3., 0.], [10., 0., 0., -20.]]]))
        result = self.runner.score_mcq(Model(), Tokenizer(), p.mcq_row('Q?', ['x','y'], 'A'))
        self.assertEqual(result['prediction'], 'A')
        self.assertLess(result['scores']['B'], -20)


if __name__ == '__main__':
    unittest.main()
