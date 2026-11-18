#!/usr/bin/env python3
"""Read-only Volume metadata inventory. No containers, deletion or remote writes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import subprocess


def group_path(path):
    parts = PurePosixPath(path.lstrip('/')).parts
    for i, part in enumerate(parts):
        if part.startswith('models--'):
            return '/'.join(parts[:i + 1])
    for i, part in enumerate(parts):
        if part.startswith('update-') and part[7:].isdigit():
            return '/'.join(parts[:i + 1])
    if parts and parts[0].startswith('understanding-adaptation-'):
        return '/'.join(parts[:min(3, len(parts))])
    return parts[0] if parts else '/'


def summarize(entries):
    groups = {}
    for entry in entries:
        counter = groups.setdefault(group_path(entry['path']), Counter())
        counter['entries'] += 1
        counter[entry['type'].lower()] += 1
        if entry['type'] == 'FILE':
            counter['logical_file_bytes'] += entry['bytes']
    return [{'path': path, **dict(counts),
             'gib': round(counts['logical_file_bytes'] / 2**30, 4)}
            for path, counts in sorted(groups.items(),
                key=lambda item: item[1]['logical_file_bytes'], reverse=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', default='createdliving1000')
    parser.add_argument('--env', default='main')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        raise ValueError('Use a new output directory or --resume')
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ.update(MODAL_PROFILE=args.profile, MODAL_ENVIRONMENT=args.env)
    import modal
    catalog_path = args.out / 'volumes.json'
    if catalog_path.exists():
        volumes = json.loads(catalog_path.read_text())
    else:
        import sys
        volumes = json.loads(subprocess.check_output(
            [sys.executable, '-m', 'modal', 'volume', 'list', '--env', args.env, '--json']))
        catalog_path.write_text(json.dumps(volumes, indent=2) + '\n')
    summaries = []
    for record in volumes:
        name = record['Name']
        if '/' in name or name in {'.', '..'}:
            raise ValueError('Invalid volume name')
        target = args.out / (name + '.json')
        if target.exists():
            value = json.loads(target.read_text())
            if value['profile'] != args.profile or value['environment'] != args.env:
                raise ValueError('Inventory workspace differs')
        else:
            entries = []
            volume = modal.Volume.from_name(name, create_if_missing=False)
            for entry in volume.iterdir('/', recursive=True):
                entries.append({'path': entry.path, 'type': entry.type.name,
                                'bytes': entry.size, 'mtime': entry.mtime})
            value = {'volume': name, 'profile': args.profile, 'environment': args.env,
                     'observed_at': datetime.now(timezone.utc).isoformat(),
                     'entries': entries, 'groups': summarize(entries)}
            temporary = target.with_suffix('.partial')
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(target)
        size = sum(e['bytes'] for e in value['entries'] if e['type'] == 'FILE')
        summary = {'volume': name, 'entries': len(value['entries']),
                   'logical_file_bytes': size, 'gib': round(size / 2**30, 4)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    report = {'observed_at': datetime.now(timezone.utc).isoformat(),
              'volumes': sorted(summaries, key=lambda v: v['logical_file_bytes'], reverse=True),
              'logical_file_bytes': sum(v['logical_file_bytes'] for v in summaries),
              'remote_compute_started': False, 'remote_writes': False,
              'deletion_performed': False,
              'limitations': 'Logical file sizes, excluding symlinks; not physical billed storage. '
                             'Daily billing snapshots, free allowance, sharing and deletion lag may differ.'}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
