#!/usr/bin/env python3
"""Offline post-assembly identity audit; never modifies either sealed input."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

import r7_public_benchmark_protection as protection
from r7_public_benchmark_protection import digest, read, require, save, sha


def literal_fields(path, variable):
    matches = [ast.literal_eval(node.value) for node in ast.walk(ast.parse(Path(path).read_text()))
               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)]
    require(len(matches) == 1 and isinstance(matches[0], set), 'Ambiguous identity field declaration')
    return matches[0]


def identities(value, fields):
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in fields:
                result.update(v for v in (child if isinstance(child, list) else [child]) if isinstance(v, str) and v)
            elif isinstance(child, (dict, list)):
                result.update(identities(child, fields))
    elif isinstance(value, list):
        for child in value:
            result.update(identities(child, fields))
    return result


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def json_lines(path, expected_sha):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            checksum.update(line)
            require(line.strip(), 'Blank witness line')
            yield json.loads(line)
    require(checksum.hexdigest() == expected_sha, 'Scanned source bytes differ from bound artifact')


def scan(witnesses, held, field_sets):
    protected = {'source_group_id': {r['source_group_id'] for r in held},
                 'source_id': {r['source_id'] for r in held},
                 'source_sha256': {r['original_ref']['source_sha256'] for r in held}}
    seen, recognized, hits = set(), {name: set() for name in field_sets}, []
    for witness in witnesses:
        require(witness['id'] not in seen, 'Duplicate witness ID')
        seen.add(witness['id'])
        matches = {}
        for name, fields in field_sets.items():
            values = identities(witness, fields)
            recognized[name].update(values)
            for kind, targets in protected.items():
                found = values & targets
                if found:
                    matches[name + ':' + kind] = sorted(found)
        values = set(strings(witness))
        for kind, targets in protected.items():
            found = values & targets
            if found:
                matches['all_string_fields:' + kind] = sorted(found)
        hashes = {token for value in values for token in re.findall(r'(?<![a-fA-F0-9])[a-fA-F0-9]{64}(?![a-fA-F0-9])', value)}
        found = {token.lower() for token in hashes} & protected['source_sha256']
        if found:
            matches['embedded_source_sha256'] = sorted(found)
        if matches:
            hits.append({'witness_id': witness['id'], 'witness_sha256': digest(witness), 'matches': matches})
    return {'witness_count': len(seen), 'witness_ids_sha256': digest(sorted(seen)),
            'protected_counts': {kind: len(values) for kind, values in protected.items()},
            'recognized_fields': {name: sorted(fields) for name, fields in field_sets.items()},
            'recognized_identity_counts': {name: len(values) for name, values in recognized.items()},
            'recognized_identity_sha256': {name: digest(sorted(values)) for name, values in recognized.items()},
            'intersection_counts': {name: {kind: len(values & targets) for kind, targets in protected.items()}
                                    for name, values in recognized.items()},
            'matching_witness_count': len(hits), 'matching_witnesses': hits,
            'zero_intersection': not hits}, seen


def audit(pack, pack_sha, held_pack, held_sha, out):
    pack, held_pack, out = Path(pack).resolve(), Path(held_pack).resolve(), Path(out).absolute()
    require(not out.exists() and not out.is_relative_to(pack) and not out.is_relative_to(held_pack), 'New separate audit directory required')
    require(sha(pack / 'manifest.json') == pack_sha and read(pack / 'PREPARED.json') == {'manifest_sha256': pack_sha}, 'Pack seal differs')
    manifest = read(pack / 'manifest.json')
    protection.verify(held_pack, held_sha)
    for name in ('source-witnesses.jsonl', 'rows.jsonl', 'report.json'):
        require(sha(pack / name) == manifest['artifacts'][name], 'Bound pack artifact differs')
    report = read(pack / 'report.json')
    require(report['input_files'][str(held_pack / 'manifest.json')] == held_sha, 'Pack did not bind this protection release')
    held = list(json_lines(held_pack / 'held-content.jsonl', read(held_pack / 'manifest.json')['artifacts']['held-content.jsonl']))
    sources = {'assembler': Path(__file__).with_name('prepare_r7_source_inventory.py'),
               'runner': Path(__file__).with_name('run_understanding_r7.py')}
    fields = {'assembler': literal_fields(sources['assembler'], 'IDENTITIES'),
              'runner': literal_fields(sources['runner'], 'identities')}
    code_hashes = {name: sha(path) for name, path in sources.items()}
    require(all('source_group_id' not in values and {'source_id', 'group_id', 'group_aliases'} <= values for values in fields.values()),
            'Readers changed; reassess rather than repeat the original field-gap finding')
    result, witness_ids = scan(json_lines(pack / 'source-witnesses.jsonl', manifest['artifacts']['source-witnesses.jsonl']), held, fields)
    row_ids = []
    for row in json_lines(pack / 'rows.jsonl', manifest['artifacts']['rows.jsonl']):
        row_ids.append(row['id'])
    require(len(row_ids) == len(set(row_ids)) == len(witness_ids) == report['retained_views']
            and set(row_ids) == witness_ids, 'Audit did not cover all retained training views')
    result.update(version='public-benchmark-protection-identity-audit-v1', pack_manifest_sha256=pack_sha,
                  protection_manifest_sha256=held_sha, source_witnesses_sha256=manifest['artifacts']['source-witnesses.jsonl'],
                  rows_sha256=manifest['artifacts']['rows.jsonl'], reader_code_sha256=code_hashes,
                  field_gap='source_group_id is not recognized by either current reader. V1 had recognized source_id and text protection only.',
                  scope='All retained pack witnesses, recursive recognized identity fields, all string fields, and embedded SHA-256 values.',
                  impact='No additional retained views would be excluded by these exact public group/source-hash aliases.' if result['zero_intersection'] else 'Intersection found; review matching witnesses before training.',
                  content_overlap_recomputed=False, foundation_unseen_claim=False,
                  limitations=['Does not establish semantic family equivalence, unknown source ancestry, or pretraining separation.'])
    out.mkdir(parents=True)
    save(out / 'audit.json', result)
    # A new recognized-field view for future assemblies, not a mutation of pack v1.
    with (out / 'held-content.jsonl').open('x', encoding='utf-8') as stream:
        for row in held:
            value = {**row, 'group_id': row['source_group_id'],
                     'group_aliases': [row['source_group_id'], row['original_ref']['source_sha256']]}
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n')
    (out / 'code').mkdir()
    for name, path in {**sources, 'audit': Path(__file__), 'v1_builder': Path(protection.__file__)}.items():
        shutil.copyfile(path, out / 'code' / (name + '.py'))
    save(out / 'pack-manifest.json', manifest)
    save(out / 'protection-manifest.json', read(held_pack / 'manifest.json'))
    release = {'version': 'public-benchmark-protection-supplement-v1', 'held_content_version': 'public-benchmark-protection-v2',
               'training_eligible': False, 'existing_pack_modified': False, 'held_units': len(held),
               'pack_manifest_sha256': pack_sha, 'protection_v1_manifest_sha256': held_sha,
               'zero_intersection': result['zero_intersection'], 'checked_witnesses': len(witness_ids),
               'artifacts': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    save(out / 'manifest.json', release)
    checksum = sha(out / 'manifest.json')
    save(out / 'SEALED.json', {'manifest_sha256': checksum})
    protection.verify(held_pack, held_sha)
    require(sha(pack / 'manifest.json') == pack_sha and all(sha(path) == code_hashes[name] for name, path in sources.items()),
            'Inputs/readers changed during audit')
    return {'path': str(out.resolve()), 'manifest_sha256': checksum, 'checked_witnesses': len(witness_ids),
            'zero_intersection': result['zero_intersection'], 'intersections': result['intersection_counts'], 'gpu_launched': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pack', 'protection', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pack-sha', required=True)
    parser.add_argument('--protection-sha', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.pack, args.pack_sha, args.protection, args.protection_sha, args.out), indent=2))


if __name__ == '__main__':
    main()
