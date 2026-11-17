"""Read-only collection of sealed benchmark evidence; never starts inference."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile

os.environ.setdefault('MODAL_PROFILE', 'createdliving1000')
os.environ.setdefault('MODAL_ENVIRONMENT', 'main')
import modal

app = modal.App('ghana-health-r6-benchmark-evidence')
volume = modal.Volume.from_name('ghana-health-understand-train', create_if_missing=False)


@app.function(image=modal.Image.debian_slim(python_version='3.11'),
              volumes={'/artifacts': volume.read_only()}, timeout=1800, retries=0)
def collect(run_id):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{8,80}', run_id):
        raise ValueError('Invalid run ID')
    root = Path('/artifacts/understanding-adaptation-r6/benchmark-corrections') / run_id
    states = {v: json.loads((root / f'{v}.json').read_text()) for v in ('base', 'r5', 'r6')}
    if any(value['state'] != 'completed' for value in states.values()):
        return {'states': states, 'complete': False}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        count = 0
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.is_symlink() and not path.name.endswith('.tmp'):
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
                count += 1
                if count % 500 == 0:
                    print(f'Archived {count} evidence files', flush=True)
    blob = stream.getvalue()
    # Re-read original aggregate files to prove this pass did not replace them.
    original = Path('/artifacts/understanding-adaptation-r6/benchmark-results')
    legacy_hashes = {v: hashlib.sha256((original / f'{v}.json').read_bytes()).hexdigest() for v in states}
    return {'complete': True, 'archive': blob, 'sha256': hashlib.sha256(blob).hexdigest(),
            'legacy_hashes_readback': legacy_hashes}


def collection_call(run_id, path):
    """Keep the CPU archive job alive through a client disconnect and reuse its result."""
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt['run_id'] != run_id:
            raise ValueError('Collection receipt belongs to a different run')
        return modal.FunctionCall.from_id(receipt['call_id'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with modal.enable_output(), app.run(detach=True):
        call = collect.spawn(run_id)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'run_id': run_id, 'call_id': call.object_id}, indent=2)+'\n')
        temporary.replace(path)
    print(f'Detached evidence collection saved in {path}', flush=True)
    return call


def paired(before, after):
    gains = sum(not a['correct'] and b['correct'] for a,b in zip(before,after))
    losses = sum(a['correct'] and not b['correct'] for a,b in zip(before,after))
    discordant = gains + losses
    pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gains,losses)+1)) / 2**discordant) if discordant else 1.
    return {'gains': gains, 'losses': losses, 'ties': len(before)-discordant,
            'delta_points': 100 * (gains-losses) / len(before),
            'mcnemar_exact_two_sided_p': pvalue,
            'inference_limit': 'descriptive source-order subset; no multiple-comparison adjustment'}


def verify(root):
    import r6_benchmark_protocol as protocol
    read = lambda path: json.loads(Path(path).read_text())
    manifest = read(root / 'manifest.json')
    manifest_sha = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    protocol.require(read(root / 'SEALED.json') == {'manifest_sha256': manifest_sha}, 'Seal mismatch')
    inputs = read(root / 'inputs.json')
    protocol.require(protocol.digest(inputs) == manifest['inputs_sha256'], 'Input mismatch')
    for path, checksum in manifest['code'].items():
        protocol.require(hashlib.sha256((root / 'code' / path).read_bytes()).hexdigest() == checksum, 'Code mismatch')
    for variant, entry in manifest['legacy'].items():
        protocol.require(hashlib.sha256((root / 'legacy' / f'{variant}.json').read_bytes()).hexdigest() == entry['sha256'], 'Legacy changed')
    evidence, result = {}, {'run_id': manifest['run_id'], 'protocol': manifest.get('protocol'),
        'manifest_sha256': manifest_sha, 'model': manifest['model'], 'revision': manifest['revision'],
        'adapters': manifest['adapters'], 'sources': manifest.get('sources', {}), 'tasks': {}}
    identities = []
    for variant in protocol.VARIANTS:
        summary = read(root / f'{variant}.json')
        protocol.require(summary['state'] == 'completed' and summary['manifest_sha256'] == manifest_sha, 'Incomplete/mismatched run')
        protocol.require(summary['model'] == manifest['model'] and summary['revision'] == manifest['revision']
                         and summary['adapter'] == manifest['adapters'].get(variant), 'Checkpoint identity mismatch')
        identities.append((summary['model'], summary['revision'], summary['runtime'],
                           summary['tokenizer_sha256'], summary['chat_template_sha256']))
        evidence[variant] = {}
        for name, rows in inputs.items():
            folder = root / 'predictions' / variant / name
            protocol.require(len(list(folder.glob('*.json'))) == len(rows), 'Wrong result count')
            outputs = []
            for row in rows:
                item = read(folder / f'{row["source_index"]:06d}.json')
                protocol.require(all(item[k] == v for k,v in row.items()) and item['variant'] == variant
                                 and item['manifest_sha256'] == manifest_sha, 'Row identity mismatch')
                prediction = protocol.classify_output(item['raw_output']) if name == 'sib200_twi' else protocol.choose(item['scores'], row['labels'])
                protocol.require(item['prediction'] == prediction and item['correct'] == (prediction == row['gold']), 'Score mismatch')
                outputs.append(item)
            measured = protocol.summarize(outputs)
            # Python 3.12 changed float summation; tolerate rounding only in the
            # derived chance baseline, never in correct/total/invalid counts.
            actual = summary['benchmarks'][name]
            protocol.require(all(math.isclose(actual[k], v, rel_tol=0, abs_tol=1e-12)
                                 if k == 'chance' else actual[k] == v
                                 for k,v in measured.items()), 'Aggregate mismatch')
            evidence[variant][name] = outputs
    protocol.require(all(identity == identities[0] for identity in identities), 'Unmatched model/runtime/tokenizer')
    for name in inputs:
        result['tasks'][name] = {v: protocol.summarize(evidence[v][name]) for v in protocol.VARIANTS}
        result['tasks'][name]['r6_vs_r5'] = paired(evidence['r5'][name], evidence['r6'][name])
        result['tasks'][name]['r6_vs_base'] = paired(evidence['base'][name], evidence['r6'][name])
        result['tasks'][name]['subjects'] = {}
        for subject in sorted({r['subject'] for r in inputs[name]}):
            result['tasks'][name]['subjects'][subject] = {
                v: protocol.summarize([r for r in evidence[v][name] if r['subject'] == subject]) for v in protocol.VARIANTS}
        if name == 'sib200_twi':
            result['tasks'][name]['by_label'] = {label: {
                v: protocol.summarize([r for r in evidence[v][name] if r['gold'] == label])
                for v in protocol.VARIANTS} for label in protocol.SIB_LABELS}
    result['verified'] = True
    return result


def main():
    import r6_benchmark_protocol as protocol
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--summary-out', type=Path)
    args = parser.parse_args()
    if not args.verify_only:
        protocol.require(not args.out.exists(), 'Do not overwrite collected evidence; use --verify-only')
        call = collection_call(args.run_id, args.out.with_name(args.out.name + '-collection-call.json'))
        receipt = call.get()
        if not receipt['complete']:
            print(json.dumps({'complete':False, 'states':{
                variant: {'state':item['state'], 'finished_tasks': list(item.get('benchmarks', {})),
                          'error': item.get('error')}
                for variant,item in receipt['states'].items()}}, indent=2))
            return
        protocol.require(not args.out.exists(), 'Do not overwrite collected evidence')
        args.out.mkdir(parents=True)
        blob = receipt.pop('archive')
        protocol.require(hashlib.sha256(blob).hexdigest() == receipt['sha256'], 'Transport checksum mismatch')
        (args.out / 'evidence.tar.gz').write_bytes(blob)
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
            archive.extractall(args.out, filter='data')
        (args.out / 'collection-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    report = verify(args.out)
    receipt = json.loads((args.out / 'collection-receipt.json').read_text())
    manifest = json.loads((args.out / 'manifest.json').read_text())
    protocol.require(all(receipt['legacy_hashes_readback'][v] == manifest['legacy'][v]['sha256']
                         for v in protocol.VARIANTS), 'Original result files changed')
    (args.out / 'verified-summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'run_id':report['run_id'], 'verified':True, 'tasks':{
        name: {key:value for key,value in metrics.items() if key not in {'subjects', 'by_label'}}
        for name,metrics in report['tasks'].items()}}, indent=2))


if __name__ == '__main__':
    main()
