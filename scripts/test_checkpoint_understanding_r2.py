"""Local archive tests; no network, model or GPU use."""
import hashlib
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import checkpoint_understanding_r2 as checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.run = self.root / 'source', self.root / 'run'
        (self.source / 'docs').mkdir(parents=True)
        (self.source / 'scripts').mkdir()
        for name in ('CONTINUE.md', 'research-data-execution.md'):
            (self.source / 'docs' / name).write_text('test documentation')
        (self.source / 'scripts/prepare_understanding_r2.py').write_text('# fixture')
        (self.source / '.env').write_text('must not be selected')
        self.run.mkdir()
        checkpoint.save(self.run / 'train.call.json', {
            'state': 'submitted', 'call_id': 'fixture-call', 'run_id': 'fixture-run'})
        self.out = self.root / 'backup'

    def freeze(self):
        with patch.object(checkpoint, 'ROOT', self.source):
            return checkpoint.freeze(self.source, self.run, self.out)

    def test_only_allowlisted_members_and_exact_bytes(self):
        manifest = self.freeze()
        self.assertFalse(any('.env' in n for n in manifest['files']))
        with tarfile.open(self.out / 'checkpoint.tar.gz') as archive:
            self.assertEqual(set(archive.getnames()), set(manifest['files']) | {'manifest.json'})
            for name, record in manifest['files'].items():
                data = archive.extractfile(name).read()
                self.assertEqual(len(data), record['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
        with self.assertRaises(ValueError):
            self.freeze()

    def test_unconfirmed_submission_refused(self):
        checkpoint.save(self.run / 'train.call.json', {'state': 'submitting', 'run_id': 'fixture-run'})
        with self.assertRaises(ValueError):
            self.freeze()
        self.assertFalse(self.out.exists())

    def test_evidence_selected_but_live_conversations_excluded(self):
        results = self.run / 'eval-results-v2' / 'fixture-run'
        results.mkdir(parents=True)
        (results / 'summary.json').write_text('{}')
        (results / 'audition.sqlite3').write_bytes(b'private conversation')
        (self.run / 'audition.sqlite3').write_bytes(b'private conversation')
        (results / '__pycache__').mkdir()
        (results / '__pycache__/generated.json').write_text('{}')
        (self.source / 'scripts/r2_inference.py').write_text('# fixture')
        manifest = self.freeze()
        self.assertIn('research/eval-results-v2/fixture-run/summary.json', manifest['files'])
        self.assertIn('r2/scripts/r2_inference.py', manifest['files'])
        self.assertFalse(any('sqlite' in name or '__pycache__' in name for name in manifest['files']))

    def test_selected_credential_material_refused(self):
        (self.source / 'docs/CONTINUE.md').write_text('-----BEGIN ' + 'PRIVATE KEY-----')
        with self.assertRaises(ValueError):
            self.freeze()
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
