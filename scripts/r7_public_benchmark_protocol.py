"""Frozen public benchmark contracts, evidence validation and denominator-aware reports."""
from collections import Counter
import importlib.util
import math
from pathlib import Path
import re
import shutil
import sys

from r7_public_benchmark_protection import COUNTS, INPUTS_SHA, SOURCE_MANIFEST_SHA
from r7_public_benchmark_protection import digest, read, require, save, sha, source_inputs

VERSION = 'r7-public-benchmark-v1'
FROZEN_VERSION = 'r7-public-benchmark-frozen-v1'
VARIANTS = ('r6_response', 'r7_response')
PROTOCOL_SHA = '444bd538ee3e7f56713e8eb29adcca3b45187476162c80a0d6c2d4bb7cdc2f63'
R2_SHA = 'c8ac9598d3108e4d93774a0dfe1cb753f5793460cc81a5fec73a5b75dac075f2'
R6_BUNDLE_SHA = '529200bb9482784bf7045985de3ffda7a712055978ec9f302f19dfeec4ed670a'
SCORING = {'mcq': 'conditional_sum_log_probability_of_presented_answer_letters',
           'sib': 'strict_single_label_generation', 'enable_thinking': False,
           'max_new_tokens': 64, 'do_sample': False, 'num_beams': 1,
           'repetition_penalty': 1.0, 'seed': 17, 'truncate': False}


def import_file(path, expected_sha=None):
    path = Path(path).resolve()
    require(expected_sha in (None, sha(path)), 'Executable bytes differ before import')
    spec = importlib.util.spec_from_file_location('_public_' + path.stem + '_' + sha(path)[:16], path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


protocol = import_file(Path(__file__).with_name('r6_benchmark_protocol.py'), PROTOCOL_SHA)


def safe(root, name):
    root, relative = Path(root), Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path')
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            'Missing or linked artifact: ' + name)
    return path


def inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')),
            'Linked artifact directory')
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def validate_run_id(run_id):
    require(isinstance(run_id, str) and re.fullmatch('r7-public-[a-z0-9-]{4,80}', run_id), 'Invalid public run ID')


def verify_historical(root, launch):
    root = Path(root)
    manifest, inputs = source_inputs(root, launch)
    for name, checksum in manifest['code'].items():
        require(sha(safe(root, 'code/' + name)) == checksum, 'Historical code differs')
    summaries = {}
    identities = []
    for variant in protocol.VARIANTS:
        require(sha(safe(root, 'legacy/' + variant + '.json')) == manifest['legacy'][variant]['sha256'],
                'Legacy result changed')
        summary = read(root / (variant + '.json'))
        require(summary['state'] == 'completed' and summary['manifest_sha256'] == SOURCE_MANIFEST_SHA
                and summary['adapter'] == manifest['adapters'].get(variant)
                and summary['model'] == manifest['model'] and summary['revision'] == manifest['revision'],
                'Historical summary identity differs')
        identities.append({key: summary[key] for key in ('model', 'revision', 'runtime',
                                                        'tokenizer_sha256', 'chat_template_sha256')})
        summaries[variant] = {}
        for name, rows in inputs.items():
            folder = root / 'predictions' / variant / name
            require({p.name for p in folder.iterdir()} == {f'{r["source_index"]:06d}.json' for r in rows},
                    'Historical prediction inventory differs')
            evidence = []
            for row in rows:
                item = read(folder / f'{row["source_index"]:06d}.json')
                require(all(item.get(k) == v for k, v in row.items()) and item['variant'] == variant
                        and item['manifest_sha256'] == SOURCE_MANIFEST_SHA, 'Historical case identity differs')
                prediction = (protocol.classify_output(item['raw_output']) if name == 'sib200_twi'
                              else protocol.choose(item['scores'], row['labels']))
                require(item['prediction'] == prediction and type(item['correct']) is bool
                        and item['correct'] == (prediction == row['gold']), 'Historical score differs')
                evidence.append(item)
            measured = protocol.summarize(evidence)
            actual = summary['benchmarks'][name]
            require(all(math.isclose(actual[k], v, rel_tol=0, abs_tol=1e-12) if k == 'chance'
                        else actual[k] == v for k, v in measured.items()), 'Historical denominator differs')
            summaries[variant][name] = measured
    require(all(identity == identities[0] for identity in identities), 'Historical runtime/tokenizer mismatch')
    return {'source_manifest_sha256': SOURCE_MANIFEST_SHA, 'cases_per_variant': 2004,
            'historical_scores_only': summaries, 'historical_execution': identities[0],
            'eligible_as_verified_r6_comparator': False, 'required_action': 'rerun_both_r6_and_r7_with_verified_native_loader',
            'loader_issue': {
                'old_runner': 'modal/train/benchmark_r6.py:load_model',
                'missing': ['explicit ^model.language_model. -> model. key_mapping',
                            'output_loading_info and strict key/shape verification',
                            'base safetensors byte/hash verification', 'loaded adapter tensor equality proof',
                            'pinned FLA/triton dependencies and native wrapper kernel-binding proof'],
                'conclusion': 'Old outputs do not establish exact student loading. This is not proof of a numerical score error.'},
            'scoring_retained': SCORING, 'foundation_unseen_claim': False}


def freeze(results, launch, out):
    results, launch, out = Path(results).resolve(), Path(launch).resolve(), Path(out).absolute()
    require(not out.exists() and not out.is_relative_to(results), 'Use a new frozen directory')
    audit = verify_historical(results, launch)
    before = inventory(results)
    out.mkdir(parents=True)
    shutil.copytree(results, out / 'historical')
    shutil.copyfile(launch, out / 'launch.json')
    require(inventory(out / 'historical') == before == inventory(results), 'Historical archive changed during freeze')
    save(out / 'loader-audit.json', audit)
    manifest = {'version': FROZEN_VERSION, 'source_manifest_sha256': SOURCE_MANIFEST_SHA,
                'inputs_sha256': INPUTS_SHA, 'counts': COUNTS, 'cases': 2004,
                'selection': 'exact_correctedR6_saved_inputs_no_new_sampling',
                'historical_comparison_allowed': False, 'training': False,
                'artifacts': inventory(out)}
    save(out / 'manifest.json', manifest)
    checksum = sha(out / 'manifest.json')
    save(out / 'SEALED.json', {'manifest_sha256': checksum})
    verify_frozen(out, checksum)
    return {'frozen': str(out.resolve()), 'manifest_sha256': checksum, 'counts': COUNTS, 'gpu_launched': False}


def verify_frozen(root, expected_sha):
    root = Path(root)
    require(sha(safe(root, 'manifest.json')) == expected_sha, 'Frozen manifest differs')
    manifest = read(root / 'manifest.json')
    require(read(root / 'SEALED.json') == {'manifest_sha256': expected_sha}, 'Frozen seal differs')
    require(manifest['version'] == FROZEN_VERSION and manifest['inputs_sha256'] == INPUTS_SHA
            and manifest['counts'] == COUNTS and manifest['historical_comparison_allowed'] is False,
            'Wrong frozen benchmark')
    actual = inventory(root)
    require(set(actual) == set(manifest['artifacts']) | {'manifest.json', 'SEALED.json'}, 'Unbound frozen artifact')
    require(all(actual[name] == checksum for name, checksum in manifest['artifacts'].items()), 'Frozen artifact changed')
    _, inputs = source_inputs(root / 'historical', root / 'launch.json')
    return manifest, inputs


def bundle_module(bundle, expected_sha):
    """Authenticate all bundle files before importing any of its runtime code."""
    bundle = Path(bundle).resolve()
    require(isinstance(expected_sha, str) and re.fullmatch('[a-f0-9]{64}', expected_sha)
            and sha(safe(bundle, 'manifest.json')) == expected_sha, 'Expected sealed completed R7 bundle')
    manifest = read(bundle / 'manifest.json')
    require(manifest['version'] == 'r7-comparison-inference-v1' and manifest['r6_bundle_sha256'] == R6_BUNDLE_SHA,
            'Completed R7 comparison bundle required; an R6-only bundle or smoke is not a checkpoint')
    require(read(bundle / 'SEALED.json') == {'manifest_sha256': expected_sha}, 'Bundle seal differs')
    files = inventory(bundle)
    require(set(files) == set(manifest['files']) | {'manifest.json', 'SEALED.json'}, 'Unbound bundle artifact')
    require(all(files[name] == checksum for name, checksum in manifest['files'].items()), 'Bundle artifact changed')
    require(manifest['files'].get('code/r2_inference.py') == R2_SHA, 'Verified native loader changed')
    # Reject cached helper modules from a different bundle before sibling imports.
    code = bundle / 'code'
    sys.dont_write_bytecode = True
    for path in code.glob('*.py'):
        cached = sys.modules.get(path.stem)
        if cached is not None and Path(getattr(cached, '__file__', '')).resolve() != path:
            del sys.modules[path.stem]
    sys.path.insert(0, str(code))
    module = import_file(safe(bundle, 'code/r7_inference.py'), manifest['files']['code/r7_inference.py'])
    verified = module.load_bundle(bundle, expected_sha)
    require(verified == manifest and set(VARIANTS) <= set(manifest['variants']), 'Bundle validation differs')
    return manifest, module


def verify_execution(package, expected_sha):
    package = Path(package)
    require(sha(safe(package, 'execution.json')) == expected_sha, 'Execution seal differs')
    execution = read(package / 'execution.json')
    require(read(package / 'PREPARED.json') == {'execution_sha256': expected_sha}, 'Preparation marker differs')
    actual = inventory(package)
    require(set(actual) == set(execution['files']) | {'execution.json', 'PREPARED.json'}, 'Unbound execution artifact')
    require(all(actual[name] == checksum for name, checksum in execution['files'].items()), 'Execution artifact changed')
    binding = execution['binding']
    validate_run_id(binding['run_id'])
    require(binding['version'] == VERSION and binding['scoring'] == SCORING
            and binding['inputs_sha256'] == INPUTS_SHA and set(binding['adapters']) == set(VARIANTS)
            and execution['detached'] is True and execution['gpu'] == 'H100', 'Wrong execution contract')
    require(binding['code'] == {name: checksum for name, checksum in actual.items() if name.startswith('code/')}
            and sha(__file__) == binding['code']['code/r7_public_benchmark_protocol.py'], 'Runtime code differs from binding')
    _, inputs = verify_frozen(package / 'frozen', binding['frozen_manifest_sha256'])
    original = read(package / 'frozen/historical/manifest.json')
    require(binding['adapters']['r6_response'] == original['adapters']['r6']
            and binding['model'] == original['model'] and binding['revision'] == original['revision'],
            'Corrected R6 student identity differs')
    return execution, inputs


def verify_loading(loading, variant, binding):
    require(loading['identity'] == {'manifest_sha256': binding['bundle_sha256'], 'variant': variant,
            'model': binding['model'], 'revision': binding['revision'], 'base_weights_sha256': binding['base_sha256'],
            'adapter_weights_sha256': binding['adapters'][variant]['files']['adapter_model.safetensors']},
            'Loaded student identity differs')
    common = loading['common']
    require(common['runtime'] == binding['runtime'] and common['tokenizer_files'] == binding['tokenizer_files']
            and common['key_mapping'] == {r'^model\.language_model\.': 'model.'}
            and common['base_weights_verified'] is True and common['no_meta_parameters'] is True
            and common['strict_loading']['strict_text_weights_complete'] is True
            and common['strict_loading']['tensors'] > 0
            and common['attention'] == 'sdpa' and common['dtype'] == 'torch.bfloat16'
            and common['device'] == 'cuda' and common['tf32'] is False
            and common['kernel_bindings'] == {
                'chunk': 'fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule',
                'recurrent': 'fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule'}
            and loading['adapter_values_exact'] is True, 'Incomplete verified native loading proof')
    return loading


def conditional_scores(prompt_ids, continuations, logprobs):
    """logprobs returns next-token values for the last N positions, in target order."""
    require(prompt_ids and continuations and all(tokens for tokens in continuations.values()), 'Empty token sequence')
    if all(len(tokens) == 1 for tokens in continuations.values()):
        targets = [tokens[0] for tokens in continuations.values()]
        values = logprobs(prompt_ids, targets, same_position=True)
        require(len(values) == len(targets), 'Wrong single-position score count')
        scores = dict(zip(continuations, values))
    else:
        scores = {}
        for label, tokens in continuations.items():
            values = logprobs(prompt_ids + tokens[:-1], tokens, same_position=False)
            require(len(values) == len(tokens), 'Wrong conditional continuation score count')
            scores[label] = sum(values)
    require(all(math.isfinite(v) and v <= 0 for v in scores.values()), 'Invalid log probabilities')
    return scores


def verify_record(record, row, variant, binding, loading):
    require(record['id'] == row['id'] and record['input_sha256'] == row['input_sha256']
            and record['variant'] == variant and record['binding_sha256'] == digest(binding)
            and record['loading_sha256'] == digest(loading), 'Case execution identity differs')
    require(record['state'] in ('scored', 'error'), 'Unexpected finalized state')
    if record['state'] == 'error':
        require(record['output'] is None and record['correct'] is False and record['error'], 'Malformed failed case')
        return record
    output = record['output']
    require(type(output['prompt_tokens']) is int and output['prompt_tokens'] > 0
            and re.fullmatch('[a-f0-9]{64}', output['prompt_ids_sha256']), 'Missing tokenized input evidence')
    if row['task'] == 'mcq':
        require(set(output['answer_token_ids']) == set(row['labels'])
                and all(type(tokens) is list and tokens and all(type(t) is int and t >= 0 for t in tokens)
                        for tokens in output['answer_token_ids'].values()), 'Invalid candidate token evidence')
        require(all(math.isfinite(v) and v <= 0 for v in output['scores'].values()), 'Invalid saved log probability')
        prediction = protocol.choose(output['scores'], row['labels'])
    else:
        require(isinstance(output['raw_output'], str) and type(output['output_ids']) is list
                and len(output['output_ids']) <= SCORING['max_new_tokens']
                and all(type(t) is int and t >= 0 for t in output['output_ids']), 'Invalid generation evidence')
        prediction = protocol.classify_output(output['raw_output'])
    require(output['prediction'] == prediction and type(record['correct']) is bool
            and record['correct'] == (prediction == row['gold']), 'Saved score differs')
    return record


def count_rows(rows, records):
    selected = [records[row['id']] for row in rows if row['id'] in records]
    scored = [r for r in selected if r['state'] == 'scored']
    correct = sum(r['correct'] for r in scored)
    return {'expected': len(rows), 'finalized': len(selected), 'scored': len(scored),
            'pending': len(rows) - len(selected), 'errors': len(selected) - len(scored),
            'invalid': sum(r['output']['prediction'] is None for r in scored), 'correct': correct,
            'accuracy_scored': correct / len(scored) if scored else None,
            'correct_over_expected': correct / len(rows) if rows else None,
            'final_accuracy': correct / len(rows) if rows and len(scored) == len(rows) else None}


def comparison(rows, by_variant):
    before, after = (by_variant[v] for v in VARIANTS)
    paired = [row for row in rows if row['id'] in before and row['id'] in after
              and before[row['id']]['state'] == after[row['id']]['state'] == 'scored']
    for row in paired:
        a, b = before[row['id']]['output'], after[row['id']]['output']
        require(a['prompt_ids_sha256'] == b['prompt_ids_sha256'] and a['prompt_tokens'] == b['prompt_tokens']
                and a.get('answer_token_ids') == b.get('answer_token_ids'), 'Paired tokenized inputs differ')
    gains = sum(not before[r['id']]['correct'] and after[r['id']]['correct'] for r in paired)
    losses = sum(before[r['id']]['correct'] and not after[r['id']]['correct'] for r in paired)
    return {**{v: count_rows(rows, by_variant[v]) for v in VARIANTS}, 'paired': {
        'expected': len(rows), 'scored_pairs': len(paired), 'unscored_pairs': len(rows) - len(paired),
        'gains': gains, 'losses': losses, 'ties': len(paired) - gains - losses,
        'paired_subset_delta_points': 100 * (gains - losses) / len(paired) if paired else None,
        'primary_delta_points': 100 * (gains - losses) / len(rows) if rows and len(paired) == len(rows) else None}}


def report(inputs, records):
    allowed = {row['id'] for rows in inputs.values() for row in rows}
    indexed = {variant: {} for variant in VARIANTS}
    for record in records:
        require(record['variant'] in indexed and record['id'] in allowed, 'Unknown reported case')
        target = indexed[record['variant']]
        require(record['id'] not in target, 'Duplicate reported case')
        target[record['id']] = record
    tasks = {}
    for name, rows in inputs.items():
        tasks[name] = comparison(rows, indexed)
        tasks[name]['subjects'] = {subject: comparison([r for r in rows if r['subject'] == subject], indexed)
                                   for subject in sorted({r['subject'] for r in rows})}
        if name == 'sib200_twi':
            tasks[name]['labels'] = {label: comparison([r for r in rows if r['gold'] == label], indexed)
                                     for label in protocol.SIB_LABELS}
    all_rows = [row for rows in inputs.values() for row in rows]
    return {'version': VERSION, 'total': comparison(all_rows, indexed), 'tasks': tasks,
            'historical_scores_used_as_comparator': False, 'foundation_unseen_claim': False,
            'limits': ['Source-order subsets, not full benchmark estimates; PubMedQA is the saved labeled train subset.',
                       'Missing/error cases stay in expected denominators; primary deltas require all pairs scored.',
                       'Gold labels and prompts are protected from R7 training; unknown prior contamination is not excluded.']}
