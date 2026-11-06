#!/usr/bin/env python3
"""Archive explicit R2 code, documentation and receipts; never launch compute."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from run_understanding_r2 import ROOT, VOLUME, read, require, save, sha


def freeze(source, run, out):
    require(not out.exists(), 'Use a fresh checkpoint directory')
    paths = {}
    for folder, pattern in (('scripts', '*understanding_r2*.py'), ('modal/train', '*understanding_r2*.py'),
                            ('docs', 'TWI_UNDERSTANDING_R2_*.md'), ('docs', 'TWI_R2_LIVE_*.md')):
        for path in (ROOT / folder).glob(pattern):
            paths['r2/' + path.relative_to(ROOT).as_posix()] = path
    for name in ('scripts/r2_inference.py', 'scripts/test_r2_inference.py', 'scripts/probe_r2_live.py',
                 'modal/research_understanding_r2_demo.py', 'scripts/r2_comparison.py',
                 'scripts/test_r2_comparison.py', 'scripts/import_r2_comparison_feedback.py'):
        if (ROOT / name).is_file():
            paths['r2/' + name] = ROOT / name
    for name in ('docs/CONTINUE.md', 'docs/research-data-execution.md'):
        paths['original/' + name] = source / name
    meeting = source / 'tmp/jason-lucas-20261106/progress-20261105.html'
    if meeting.is_file():
        paths['original/' + meeting.relative_to(source).as_posix()] = meeting
    screenshot = source / 'tmp/jason-lucas-20261106/r2-live-20261105.jpg'
    if screenshot.is_file():
        paths['original/' + screenshot.relative_to(source).as_posix()] = screenshot
    comparison_shot = source / 'tmp/jason-lucas-20261106/r2-comparison-ui-20261106.jpg'
    if comparison_shot.is_file():
        paths['original/' + comparison_shot.relative_to(source).as_posix()] = comparison_shot
    report = source / 'tmp/jason-lucas-20261106/r2-results-v2'
    if report.is_dir():
        for path in report.iterdir():
            if path.is_file() and path.suffix in {'.json', '.html'}:
                paths['original/' + path.relative_to(source).as_posix()] = path
    # Only research artifacts, never the live conversation/review SQLite store.
    for folder in (run / 'eval-bundle-v2', run / 'eval-results-v2', run / 'live-probe-v1'):
        if folder.is_dir():
            for path in folder.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.json', '.py', '.jinja'}:
                    paths['research/' + path.relative_to(run).as_posix()] = path
    for path in run.glob('*.json'):
        paths['receipts/' + path.name] = path
    for folder in (run / 'smoke-v2-progress-02', *sorted(run.glob('train-progress-*'))):
        if folder.is_dir() and not folder.is_symlink():
            for path in folder.glob('*.json'):
                paths['receipts/' + path.relative_to(run).as_posix()] = path
    for name in ('pack/manifest.json', 'evaluation/manifest.json'):
        if (run / name).is_file():
            paths['receipts/' + name] = run / name
    require(paths and (run / 'train.call.json').is_file(), 'Training submission receipt required')
    receipt = read(run / 'train.call.json')
    require(receipt.get('call_id') and receipt.get('state') == 'submitted', 'No confirmed submission')
    secret = re.compile(rb'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|\b(?:hf_|ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{28,}')
    payload = {}
    for name, path in sorted(paths.items()):
        require(not path.is_symlink() and path.is_file(), 'Linked or missing source')
        data = path.read_bytes()
        require(not secret.search(data), 'Potential credential in selected file: ' + name)
        payload[name] = data
    manifest = {'schema': 'understanding-r2-checkpoint-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'private': True, 'gpu_launched': False, 'run_id': receipt['run_id'], 'call_id': receipt['call_id'],
        'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in payload.items()},
        'scope': 'Selected R2 source/tests/docs/receipts. Full frozen inputs and model checkpoints live separately in the private Modal run directory.',
        'secret_screen_limit': 'Explicit source allowlist and credential-pattern check, not a comprehensive detector.'}
    out.mkdir(parents=True)
    save(out / 'manifest.json', manifest, exclusive=True)
    payload['manifest.json'] = (out / 'manifest.json').read_bytes()
    with tarfile.open(out / 'checkpoint.tar.gz', 'w:gz') as bundle:
        for name, data in payload.items():
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(data), 0o600, 0
            bundle.addfile(item, io.BytesIO(data))
    for name, path in paths.items():
        require(sha(path) == manifest['files'][name]['sha256'], 'Source changed during checkpoint')
    return manifest


def upload(out):
    import modal
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    archive = out / 'checkpoint.tar.gz'
    prefix = '/research-training-r2-20261105/' + sha(archive)
    for name in ('checkpoint.tar.gz', 'manifest.json'):
        path = out / name
        remote = prefix + '/' + name
        try:
            digest = hashlib.sha256()
            for chunk in volume.read_file(remote):
                digest.update(chunk)
        except (FileNotFoundError, modal.exception.NotFoundError):
            with volume.batch_upload(force=False) as batch:
                batch.put_file(path, remote)
        else:
            require(digest.hexdigest() == sha(path), 'Existing remote checkpoint differs')
        digest, size = hashlib.sha256(), 0
        for chunk in volume.read_file(remote):
            digest.update(chunk)
            size += len(chunk)
        require(digest.hexdigest() == sha(path) and size == path.stat().st_size, 'Complete remote readback failed')
    manifest = read(out / 'manifest.json')
    result = {'provider': 'modal', 'volume': VOLUME, 'prefix': prefix, 'private': True,
        'archive_sha256': sha(archive), 'manifest_sha256': sha(out / 'manifest.json'),
        'full_remote_readback_verified': True, 'files': len(manifest['files']),
        'gpu_launched': False, 'verified_at': datetime.now(timezone.utc).isoformat()}
    save(out / 'storage.json', result, exclusive=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    result = freeze(args.source.resolve(), args.run.resolve(), args.out.resolve())
    if args.upload:
        result = upload(args.out.resolve())
    print(json.dumps(result, indent=2))
