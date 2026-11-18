#!/usr/bin/env python3
"""Offline, lossless protection of the exact corrected R6 public benchmark."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

VERSION = 'public-benchmark-protection-v1'
SOURCE_MANIFEST_SHA = '6f703d7aefb211c11ead4b3d6ebed50b4490f2d31353fa2f129d12d5cd5616b5'
INPUTS_SHA = 'ed2dfc9d45014348ca932e8d4af21abb722dbe8c2f9aebd65d905671cf18ec70'
COUNTS = {'afrimmlu_twi': 300, 'sib200_twi': 204, 'mmlu': 300, 'hellaswag': 300,
          'arc_challenge': 300, 'medqa': 300, 'pubmedqa': 300}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                       parse_constant=lambda _: require(False, 'Nonfinite JSON'))
    digest(value)
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Linked output')
    temporary = path.with_name(path.name + '.tmp')
    require(not temporary.is_symlink(), 'Linked temporary output')
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_inputs(root, launch):
    root, launch = Path(root), Path(launch)
    require(sha(root / 'manifest.json') == SOURCE_MANIFEST_SHA, 'Not the pinned corrected R6 manifest')
    require(read(root / 'SEALED.json') == {'manifest_sha256': SOURCE_MANIFEST_SHA}, 'Source seal differs')
    manifest, inputs, receipt = read(root / 'manifest.json'), read(root / 'inputs.json'), read(launch)
    require(manifest['protocol'] == 'r6-benchmark-v2' and receipt['run_id'] == manifest['run_id']
            and receipt['manifest_sha256'] == SOURCE_MANIFEST_SHA and receipt['counts'] == COUNTS,
            'Source launch identity/counts differ')
    require(digest(inputs) == manifest['inputs_sha256'] == INPUTS_SHA, 'Frozen public inputs changed')
    require({name: len(rows) for name, rows in inputs.items()} == COUNTS, 'Expected all 2004 public cases')
    seen = set()
    for name, rows in inputs.items():
        require(manifest['sources'][name]['selected'] == len(rows), 'Source count changed')
        for index, row in enumerate(rows):
            require(row['id'] == f'{name}:{index}' and row['source_index'] == index
                    and row['id'] not in seen, 'Source ordering or identity changed')
            require(row['input_sha256'] == digest({k: v for k, v in row.items() if k != 'input_sha256'}),
                    'Case checksum differs')
            seen.add(row['id'])
    return manifest, inputs


def held_rows(manifest, inputs):
    for benchmark, rows in inputs.items():
        source = manifest['sources'][benchmark]
        namespace = ':'.join(str(source[key]) for key in ('repo', 'revision', 'config', 'split'))
        for row in rows:
            group = 'public-source:' + namespace + ':sha256:' + row['source_sha256']
            yield {'id': row['id'], 'case_ids': [row['id']], 'benchmark': benchmark,
                   'source_id': 'public-source:' + namespace + ':row:' + str(row['source_index']),
                   'source_group_id': group, 'subject': row['subject'], 'text': row['prompt'],
                   'prompt': row['prompt'], 'options': row.get('options', []),
                   'labels': row.get('labels', []), 'gold': row['gold'],
                   'gold_use': 'protection_and_evaluation_only_never_training',
                   'training_eligible': False, 'original_ref': {
                       **{key: source[key] for key in ('repo', 'revision', 'config', 'split')},
                       **{key: row[key] for key in ('source_index', 'source_sha256', 'input_sha256')},
                       'corrected_r6_manifest_sha256': SOURCE_MANIFEST_SHA}}


def prepare(results, launch, out):
    results, launch, out = Path(results).resolve(), Path(launch).resolve(), Path(out).absolute()
    require(not out.exists() and not out.is_relative_to(results), 'Use a new protection directory outside the source')
    manifest, inputs = source_inputs(results, launch)
    rows = list(held_rows(manifest, inputs))
    out.mkdir(parents=True)
    original = out / 'original'
    original.mkdir()
    for name in ('inputs.json', 'manifest.json', 'SEALED.json'):
        shutil.copyfile(results / name, original / name)
    shutil.copyfile(launch, original / 'launch.json')
    with (out / 'held-content.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    value = {'version': VERSION, 'training_eligible': False, 'model_evaluation_executed': False,
             'counts': COUNTS, 'held_units': len(rows),
             'source_groups': len({row['source_group_id'] for row in rows}),
             'corrected_r6_manifest_sha256': SOURCE_MANIFEST_SHA, 'inputs_sha256': INPUTS_SHA,
             'source_selection': 'exact_saved_correctedR6_inputs_no_network_no_resampling',
             'group_policy': 'Identical original source-row hashes share a group within repo/revision/config/split. '
                             'All derived views must retain that group; broader document/family IDs were not saved.',
             'foundation_unseen_claim': False, 'prior_training_unseen_claim': False,
             'limitations': ['Protection prevents current known content/source reuse, not unknown pretraining contamination.',
                            'Historical source-order subsets are not full benchmark or population estimates.',
                            'PubMedQA uses its labeled train split as recorded in corrected R6.'],
             'builder_sha256': sha(__file__),
             'artifacts': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    save(out / 'manifest.json', value)
    checksum = sha(out / 'manifest.json')
    save(out / 'SEALED.json', {'manifest_sha256': checksum})
    verify(out, checksum)
    return {'path': str(out.resolve()), 'manifest_sha256': checksum, 'held_units': len(rows),
            'counts': COUNTS, 'gpu_launched': False}


def verify(out, expected_sha=None):
    out = Path(out)
    require(not out.is_symlink() and not any(p.is_symlink() for p in out.rglob('*')), 'Linked protection artifact')
    checksum = sha(out / 'manifest.json')
    require(expected_sha in (None, checksum), 'Protection manifest differs')
    require(read(out / 'SEALED.json') == {'manifest_sha256': checksum}, 'Protection seal differs')
    manifest = read(out / 'manifest.json')
    require(manifest['version'] == VERSION and manifest['training_eligible'] is False, 'Wrong protection release')
    require({p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
            == set(manifest['artifacts']) | {'manifest.json', 'SEALED.json'}, 'Unbound protection artifact')
    for name, checksum in manifest['artifacts'].items():
        path = (out / name).resolve()
        require(path.is_relative_to(out.resolve()) and sha(path) == checksum, 'Protection artifact changed')
    original, inputs = source_inputs(out / 'original', out / 'original/launch.json')
    actual = [json.loads(line) for line in (out / 'held-content.jsonl').read_text(encoding='utf-8').splitlines()]
    require(actual == list(held_rows(original, inputs)) and len(actual) == manifest['held_units'] == 2004,
            'Protection content differs from exact corrected R6')
    require(dict(Counter(row['benchmark'] for row in actual)) == manifest['counts'] == COUNTS,
            'Protection denominators differ')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--launch', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--expected-sha')
    args = parser.parse_args()
    if args.verify:
        verify(args.out, args.expected_sha)
        result = {'verified': True, 'path': str(args.out.resolve()), 'manifest_sha256': sha(args.out / 'manifest.json')}
    else:
        if not args.results or not args.launch:
            parser.error('--results and --launch required')
        result = prepare(args.results, args.launch, args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
