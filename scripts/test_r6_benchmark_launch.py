from types import SimpleNamespace
import unittest

from benchmark_launch import terminal_state


def node(state, children=()):
    return SimpleNamespace(status=SimpleNamespace(name=state), children=list(children))


class ResumeTest(unittest.TestCase):
    def test_completed_call_is_not_relaunched(self):
        self.assertEqual(terminal_state([node('SUCCESS')]), 'SUCCESS')

    def test_terminal_failure_can_be_resumed(self):
        self.assertEqual(terminal_state([node('TIMEOUT')]), 'FAILED')

    def test_empty_graph_cannot_authorize_duplicate(self):
        with self.assertRaisesRegex(ValueError, 'No terminal'):
            terminal_state([])

    def test_pending_call_cannot_authorize_duplicate(self):
        with self.assertRaisesRegex(ValueError, 'pending'):
            terminal_state([node('PENDING')])

    def test_running_child_cannot_authorize_duplicate(self):
        with self.assertRaisesRegex(ValueError, 'pending'):
            terminal_state([node('SUCCESS', [node('RUNNING')])])


if __name__ == '__main__':
    unittest.main()
