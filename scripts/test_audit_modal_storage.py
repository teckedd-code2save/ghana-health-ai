import unittest
from audit_modal_storage import group_path, summarize


class StorageAuditTests(unittest.TestCase):
    def test_model_and_checkpoint_groups_are_disjoint_prefixes(self):
        self.assertEqual(group_path('hf/hub/models--Qwen--Model/blobs/hash'), 'hf/hub/models--Qwen--Model')
        self.assertEqual(group_path('run/attempt/update-00000025/state.pt'), 'run/attempt/update-00000025')

    def test_only_regular_file_bytes_are_summed(self):
        groups = summarize([{'path': 'cache/blob', 'type': 'FILE', 'bytes': 12},
                            {'path': 'cache/link', 'type': 'SYMLINK', 'bytes': 12},
                            {'path': 'cache/dir', 'type': 'DIRECTORY', 'bytes': 99}])
        self.assertEqual(groups[0]['logical_file_bytes'], 12)
        self.assertEqual(groups[0]['entries'], 3)


if __name__ == '__main__':
    unittest.main()
