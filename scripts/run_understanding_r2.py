#!/usr/bin/env python3
"""Immutable R2 packages and explicitly authorized, non-retrying Modal submission."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'understanding-adaptation-r2'
RUNTIME = {'accelerate': '1.15.0', 'flash-linear-attention': '0.5.2', 'numpy': '2.4.2',
           'peft': '0.20.0', 'torch': '2.13.0', 'transformers': '5.17.0', 'triton': '3.7.1'}
CODE = ('scripts/prepare_understanding_r2.py', 'scripts/run_understanding_r2.py',
        'scripts/prepare_understanding_r2_evaluation.py', 'modal/train/train_understanding_r2.py')
FROZEN = {
 'modal/train/adapter_checkpoint.py': '917a1fee5501d480c803d301f2e11908fc897802f071dd7c2d82295e1b53226c',
 'modal/train/afrique_instruction_baseline_core.py': 'a382d3a291469c8186b24e95bfd6ac2edaa5bcfb6ba314d252c5409464df0359',
 'modal/train/afrique_response_core.py': '457ef298dbecf76ec62ab1cb74e5789224965bfefc5cbeedfc0bfc5b8e8a8924',
 'modal/train/medical_pilot_core.py': '49351d4aaad9f910af12835a60e735b61db9acbab20fd3383770bf532e7b2224',
 'modal/train/native_adaptation_run_core.py': 'ed7257ec1e782ce7c6bd898b1ac9385c7a1af70173adc07f5dfae8536905a49e',
 'modal/train/understanding_training_core.py': '383fa81a242ba9a96c16b8c187d4af81ae364dfc0f2809af5b7cb7af7326613d',
 'scripts/check_afrique_training_contract.py': 'efa03c36888755e65c72fefe7017cf5314df02672f1cfdf3f7bb632b37689cb8',
 'scripts/prepare_understanding_run.py': 'ea0ede9683e7e80cd22b60ddde12f0a12b0e67f490cadc0b88062bcf37f405b0',
 'scripts/response_resume_state.py': 'fa06c8a3257b20ed18eb31aac9fd7f122eb5bb8c02d69545bf5ef2bd1cb4d945',
 'scripts/training_data_inputs.py': 'f78bc7a0b07657e531122a9cdb384a1fc8e36c714004e912d2c600b4c797b0cc',
 'scripts/understanding_recovery.py': 'd9c9b4040202b88f4e05d159b97ac901c91b10bcce7220caa8b7a41fb32bd74f',
 'scripts/collect_understanding_progress.py': '52fc4e467e7ddd11ecfbfb24a6df2404ae201da54b2bb81daa2023308ea36dca',
}
VOLUME = 'ghana-health-understand-train'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'Nonfinite JSON'))


def save(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if exclusive:
        with path.open('xb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
    else:
        fd, name = tempfile.mkstemp(prefix='.receipt-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def run_id(value):
    # Retain the proven recovery helper's ID format in a separate R2 root/dictionary.
    require(isinstance(value, str) and re.fullmatch(r'understanding_\d{8}T\d{6}Z', value), 'Invalid run ID')
    return value


def safe_file(root, relative):
    root, relative = Path(root).resolve(strict=True), Path(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts, 'Unsafe relative path')
    path = root / relative
    require(path.resolve() == path and path.is_file(), 'Missing or linked package file: ' + str(relative))
    return path


def activate(source):
    source = Path(source).resolve(strict=True)
    os.environ['GHA_R2_SOURCE_ROOT'] = str(source)
    for name, checksum in FROZEN.items():
        require(sha(safe_file(source, name)) == checksum, 'Proven R1 helper changed: ' + name)
    sys.path[:0] = [str(source / 'scripts'), str(source / 'modal/train')]


def pack_data(pack, source):
    activate(source)
    from prepare_understanding_r2 import verify_pack
    manifest, rows, stages = verify_pack(Path(pack))
    require([s['name'] for s in stages] == ['language', 'meaning', 'responses'], 'Expected three ordered stages')
    for key, expected in {'learning_rate': 5e-5, 'grad_accum': 16, 'warmup_fraction': .05,
                          'checkpoint_every': 25, 'seed': 42, 'precision': 'bf16_base_fp32_adapters',
                          'packing': False, 'truncate': False, 'reset_optimizer_between_stages': True}.items():
        require(manifest['optimization'][key] == expected, 'Changed R1 optimization recipe: ' + key)
    for stage in stages:
        require(stage['schedule'] and all(type(i) is int and 0 <= i < len(rows) for i in stage['schedule']), 'Invalid schedule')
    return manifest, rows, stages


def evaluation_pack(folder, training_pack):
    from prepare_understanding_r2_evaluation import verify_pack
    manifest, cases = verify_pack(Path(folder), training_pack=Path(training_pack))
    require(cases and manifest['bindings']['training_manifest_sha256'] == sha(Path(training_pack) / 'manifest.json'),
            'Evaluation is not bound to the R2 training data')
    require(not any(p.is_symlink() for p in Path(folder).rglob('*')), 'Linked evaluation artifact')
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def compare_smoke(smoke_dir, contract):
    smoke_dir = Path(smoke_dir)
    old, status, smoke = (read(smoke_dir / n) for n in ('execution.json', 'status.json', 'smoke.json'))
    require(old['mode'] == 'smoke' and status['state'] == 'completed' and status['mode'] == 'smoke'
            and status['execution_sha256'] == sha(smoke_dir / 'execution.json'), 'Smoke execution/status mismatch')
    for key in ('version', 'data_sha256', 'runtime', 'frozen_code_hashes', 'code_hashes', 'evaluation', 'expected_stages'):
        require(old[key] == contract[key], 'Training differs from real-weight smoke: ' + key)
    require(status['data_sha256'] == contract['data_sha256'] and smoke['passed'] is True
            and smoke['real_weight_backward'] is True and smoke['full_state_recovery']['equivalent'] is True
            and smoke['resume_parameter_and_buffer_bitwise_equal'] is True
            and smoke['native_encoding_kinds'] == ['language', 'chat'], 'Incomplete real-weight smoke proof')
    run_id(status['run_id'])
    require(status['remote_directory'] == '/artifacts/' + VERSION + '/' + status['run_id'], 'Smoke is outside R2')
    return {'run_id': status['run_id'], 'execution_sha256': sha(smoke_dir / 'execution.json'),
            'smoke_sha256': sha(smoke_dir / 'smoke.json'), 'status_sha256': sha(smoke_dir / 'status.json')}


def prepare(pack, out, mode, evaluation, smoke_directory=None):
    require(mode in {'smoke', 'train'}, 'Unknown mode')
    source = Path(os.environ['GHA_R2_SOURCE_ROOT']).resolve(strict=True)
    manifest, _, stages = pack_data(pack, source)
    out, evaluation = Path(out), Path(evaluation)
    if evaluation.is_file():
        require(evaluation.name == 'manifest.json', 'Pass the frozen evaluation directory or its manifest.json')
        evaluation = evaluation.parent
    require(not out.exists(), 'Execution package is immutable; use a fresh directory')
    eval_files = evaluation_pack(evaluation, pack)
    contract = {'version': VERSION, 'mode': mode, 'data_sha256': sha(Path(pack) / 'manifest.json'),
        'runtime': RUNTIME, 'frozen_code_hashes': FROZEN, 'code_hashes': {n: sha(ROOT / n) for n in CODE},
        'soft_seconds': 780 if mode == 'smoke' else 14100, 'hard_seconds': 900 if mode == 'smoke' else 14400,
        'maximum_gpu_workers': 1, 'automatic_retries': 0,
        'resources': {'gpu': 'H100', 'cpu': 4, 'memory_mib': 49152},
        'evaluation': {'sha256': sha(evaluation / 'manifest.json'), 'files': eval_files},
        'expected_stages': [{k: v for k, v in s.items() if k != 'schedule'} for s in stages],
        'private_research_only': True, 'production_promoted': False,
        'estimated_requested_resource_cost_usd_at_hard_limit': (900 if mode == 'smoke' else 14400) * (.001097 + 4 * .0000131 + 48 * .00000222),
        'pricing_reference': 'https://modal.com/pricing', 'pricing_checked': '2026-11-05',
        'budget_basis': {'r1_actual_seconds': 6309.737, 'r1_scheduled_sequences': 14673,
            'rough_seconds_per_sequence': 6309.737 / 14673,
            'rough_r2_seconds': sum(s['scheduled_sequences'] for s in stages) * 6309.737 / 14673,
            'limits': 'Planning estimate, not measured R2 throughput; sequence lengths and checkpoint costs differ. No automatic continuation.'},
        'initialization_seed': 42, 'rng_reset_between_stages': False,
        'cost_note': 'Per invocation resource estimate, not account cap; excludes builds/storage. No automatic continuation.',
        'initialization': 'fresh_pinned_base_plus_fresh_lora_not_R1_adapter'}
    if mode == 'train':
        require(smoke_directory is not None, 'Completed real-weight smoke is required')
        contract['smoke'] = compare_smoke(smoke_directory, contract)
    out.mkdir(parents=True)
    for root, names, destination in ((source, FROZEN, 'frozen'), (ROOT, CODE, 'r2')):
        for name in names:
            target = out / destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(root, name), target)
    require(not any(p.is_symlink() for p in Path(pack).rglob('*')), 'Linked data package')
    shutil.copytree(pack, out / 'inputs')
    shutil.copytree(evaluation, out / 'evaluation')
    save(out / 'execution.json', contract, exclusive=True)
    verify_execution(out)
    evaluation_pack(out / 'evaluation', out / 'inputs')
    return contract


def verify_execution(package):
    package = Path(package)
    value = read(package / 'execution.json')
    require(value['version'] == VERSION and value['runtime'] == RUNTIME and value['frozen_code_hashes'] == FROZEN, 'Execution identity differs')
    require(value['initialization_seed'] == 42 and value['rng_reset_between_stages'] is False, 'Initialization RNG contract changed')
    require(value['mode'] in {'smoke', 'train'} and value['maximum_gpu_workers'] == 1
            and value['automatic_retries'] == 0 and value['resources'] == {'gpu': 'H100', 'cpu': 4, 'memory_mib': 49152}, 'Unsafe bounds')
    require((value['soft_seconds'], value['hard_seconds']) == ((780, 900) if value['mode'] == 'smoke' else (14100, 14400)), 'Budget changed')
    require(set(value['code_hashes']) == set(CODE), 'Incomplete R2 code contract')
    for group, entries in (('frozen', FROZEN), ('r2', value['code_hashes'])):
        for name, checksum in entries.items():
            require(sha(safe_file(package / group, name)) == checksum, 'Packaged code changed: ' + name)
    require(sha(package / 'evaluation/manifest.json') == value['evaluation']['sha256'], 'Frozen evaluation changed')
    actual = {p.relative_to(package / 'evaluation').as_posix(): sha(p) for p in (package / 'evaluation').rglob('*') if p.is_file()}
    require(actual == value['evaluation']['files'], 'Incomplete/changed evaluation package')
    require(sha(package / 'inputs/manifest.json') == value['data_sha256'], 'Training data changed')
    evaluation_pack(package / 'evaluation', package / 'inputs')
    return value


def mounted_root(mount):
    anchor = Path(mount).resolve(strict=True)
    root = anchor / VERSION
    require(not root.is_symlink(), 'Linked R2 run root')
    root.mkdir(exist_ok=True)
    require(root.resolve() == root, 'Noncanonical R2 root')
    return root


def terminal_pause(claim, outcome, identifier, execution_sha, data_sha):
    require(claim.get('execution_sha256') == execution_sha and isinstance(claim.get('call_id'), str), 'Previous claim mismatch')
    require(isinstance(outcome, dict) and outcome.get('state') == 'paused_budget'
            and outcome.get('mode') == 'train' and outcome.get('training_completed') is False
            and outcome.get('run_id') == run_id(identifier) and outcome.get('execution_sha256') == execution_sha
            and outcome.get('data_sha256') == data_sha, 'Only confirmed graceful pause can resume; never auto-restart')
    return {'call_id': claim['call_id'], 'state': 'completed'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checkpoint_proof(path, execution_sha, data_sha):
    path = Path(path)
    require(path.resolve() == path and {p.name for p in path.iterdir()} == {'COMMITTED.json', 'manifest.json', 'state.pt'}, 'Unsafe checkpoint')
    for name in ('COMMITTED.json', 'manifest.json', 'state.pt'):
        safe_file(path, name)
    manifest, seal = read(path / 'manifest.json'), read(path / 'COMMITTED.json')
    contract = read(safe_file(path.parent.parent, 'contract.json'))
    require(contract['run_sha256'] == execution_sha and contract['data_sha256'] == data_sha, 'Checkpoint stage execution differs')
    require(seal == {'manifest_sha256': sha(path / 'manifest.json')}
            and manifest['identity'] == {'run_sha256': digest(contract), 'data_sha256': data_sha}
            and sha(path / 'state.pt') == manifest['state_sha256']
            and (path / 'state.pt').stat().st_size == manifest['state_bytes'], 'Uncommitted/changed checkpoint')
    return {'directory': str(path), 'state_sha256': manifest['state_sha256'],
            'manifest_sha256': seal['manifest_sha256'], 'contract_sha256': sha(path.parent.parent / 'contract.json')}


def check_stage(record, stage):
    require(record['stage'] == stage['name'] and record['status'] == 'completed'
            and record['position']['completed_updates'] == stage['optimizer_updates']
            and record['position']['completed_microbatches'] == len(stage['schedule'])
            and record['position']['microbatches_in_update'] == 0
            and record['position']['exposure']['effective_target_tokens'] == stage['effective_target_tokens']
            and record['position']['exposure']['input_tokens'] == stage['sequence_tokens'], 'Completed stage accounting differs')


def publish_adapter(path, checkpoint, execution_sha, data_sha):
    path = Path(path)
    names = {p.name for p in path.iterdir()}
    require({'adapter_config.json', 'adapter_model.safetensors'} <= names <=
            {'adapter_config.json', 'adapter_model.safetensors', 'README.md'}, 'Unexpected adapter files')
    from check_afrique_training_contract import MODEL, REVISION
    from prepare_understanding_run import TARGET_MODULES
    config = read(safe_file(path, 'adapter_config.json'))
    require(config['base_model_name_or_path'] == MODEL and config['revision'] == REVISION
            and config['peft_type'] == 'LORA' and config['r'] == 16 and config['lora_alpha'] == 32
            and config['lora_dropout'] == .05 and set(config['target_modules']) == set(TARGET_MODULES), 'Adapter config differs')
    return {'stage': path.name.removesuffix('-adapter'), 'directory': str(path),
            'files': {name: sha(safe_file(path, name)) for name in sorted(names)},
            'checkpoint': checkpoint_proof(checkpoint, execution_sha, data_sha)}


def verify_adapter_tensors(model, records):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from response_resume_state import _read_checkpoint
    for record in records:
        checkpoint = Path(record['checkpoint']['directory'])
        state = _read_checkpoint(checkpoint, read(checkpoint / 'manifest.json')['identity'])
        expected = get_peft_model_state_dict(model, state_dict=state['model']['parameters'], save_embedding_layers=False)
        actual = load_file(str(Path(record['directory']) / 'adapter_model.safetensors'), device='cpu')
        require(expected.keys() == actual.keys() and all(torch.equal(expected[k].cpu(), actual[k]) for k in expected),
                'Adapter differs from full-state checkpoint')
        record['checkpoint_adapter_tensors_verified'] = True
        del state, expected, actual


def inherit_prefix(previous, destination, stages, selection, execution):
    """Preserve committed states and prior exports before model loading can fail."""
    execution_sha, data_sha = sha(previous / 'execution.json'), execution['data_sha256']
    require(sha(previous / 'inputs/manifest.json') == data_sha, 'Previous inputs changed')
    records, exports = read(previous / 'stages.json'), read(previous / 'adapters.json')
    require(records == read(previous / 'status.json')['stages'], 'Previous stage journal changed')
    inherited, copied_exports = [], []
    for index in range(selection['stage_index'] + 1):
        stage = stages[index]
        complete = index < selection['stage_index']
        if complete:
            check_stage(records[index], stage)
            require(records[index]['run_sha256'] == execution_sha and records[index]['data_sha256'] == data_sha, 'Inherited stage identity differs')
        source = Path(records[index]['checkpoint']) if complete else Path(selection['checkpoint'])
        require(source.resolve() == source and source.parents[2] == previous
                and source.parent.parent.name == stage['name'], 'Checkpoint outside expected previous stage')
        proof = checkpoint_proof(source, execution_sha, data_sha)
        target = destination / stage['name'] / 'attempt-inherited' / source.name
        target.parent.mkdir(parents=True)
        shutil.copyfile(source.parent.parent / 'contract.json', target.parent.parent / 'contract.json')
        shutil.copytree(source, target)
        require(checkpoint_proof(target, execution_sha, data_sha)['state_sha256'] == proof['state_sha256'], 'Inherited full state changed')
        if complete:
            record = {**records[index], 'checkpoint': str(target), 'attempt': str(target.parent),
                      'steps_path': str(target.parent / 'steps.jsonl'),
                      'inherited_from': {'run_id': previous.name, 'checkpoint': str(source)}}
            if (source.parent / 'steps.jsonl').is_file():
                shutil.copyfile(safe_file(source.parent, 'steps.jsonl'), target.parent / 'steps.jsonl')
            save(target.parent / 'report.json', record)
            save(target.parent / 'COMPLETE.json', record)
            inherited.append(record)
            source_adapter, target_adapter = previous / (stage['name'] + '-adapter'), destination / (stage['name'] + '-adapter')
            expected = next(e for e in exports if e['stage'] == stage['name'])
            require(source_adapter.resolve() == source_adapter, 'Linked inherited adapter')
            require(publish_adapter(source_adapter, source, execution_sha, data_sha)['files'] == expected['files'], 'Prior adapter export changed')
            shutil.copytree(source_adapter, target_adapter)
            copied = publish_adapter(target_adapter, target, execution_sha, data_sha)
            require(copied['files'] == expected['files'], 'Inherited adapter copy changed')
            copied['inherited_from'] = {'run_id': previous.name, 'directory': str(source_adapter)}
            copied_exports.append(copied)
        else:
            selection = {**selection, 'source_checkpoint': str(source), 'checkpoint': str(target)}
    return inherited, copied_exports, selection


def progress(receipt, volume):
    """Reuse the frozen small-metadata checker, with R2 root and all three plans."""
    package = Path(receipt['package'])
    require(sha(package / 'execution.json') == receipt['execution_sha256'], 'Progress execution changed')
    execution = read(package / 'execution.json')
    activate(package / 'frozen')
    from collect_understanding_progress import small, listing, stage_report, METRICS
    root = '/' + VERSION + '/' + run_id(receipt['run_id'])
    raw = small(volume, root + '/status.json', optional=True)
    result = {'version': 'understanding-r2-progress-v1', 'collected_at': datetime.now(timezone.utc).isoformat(),
              'run_id': receipt['run_id'], 'execution_sha256': receipt['execution_sha256']}
    if raw is None:
        return {**result, 'state': 'awaiting_remote_metadata', 'identity_verified': False, 'stages': []}
    state = json.loads(raw)
    require(state['run_id'] == receipt['run_id'] and state['execution_sha256'] == receipt['execution_sha256']
            and state['data_sha256'] == execution['data_sha256'], 'Progress status identity differs')
    remote_execution = small(volume, root + '/execution.json')
    require(hashlib.sha256(remote_execution).hexdigest() == receipt['execution_sha256'], 'Progress remote execution changed')
    schedules = read(package / 'inputs/stages.json')
    require([{k: v for k, v in s.items() if k != 'schedule'} for s in schedules] == execution['expected_stages'], 'Progress schedule changed')
    entries, stages = listing(volume, root), []
    for stage in schedules:
        plan = {**stage, 'checkpoint_schedule_sha256': digest({'indices': stage['schedule']})}
        paths = sorted(p for p in entries if re.fullmatch(re.escape(root + '/' + stage['name']) + r'/attempt-[A-Za-z0-9_-]+/report\.json', p))
        attempts = [stage_report(volume, root, p, plan, receipt['execution_sha256'], execution['data_sha256'], FROZEN, entries) for p in paths]
        if attempts:
            best = max(attempts, key=lambda x: ((x.get('committed') or {}).get('updates', -1), (x.get('observed') or {}).get('updates', -1)))
            stages.append({'name': stage['name'], **best})
        else:
            stages.append({'name': stage['name'], 'state': 'no_report_visible', 'planned': dict(zip(METRICS,
                (stage['optimizer_updates'], stage['scheduled_sequences'], stage['sequence_tokens'], stage['effective_target_tokens']))),
                'committed': None, 'observed': None})
    return {**result, 'state': state['state'], 'identity_verified': True, 'active_stage': state.get('stage'),
        'status_elapsed_seconds_at_last_write': state.get('elapsed_seconds'), 'stages': stages,
        'projection_note': 'Measured ranges are stage-only, not promises or a whole-run ETA. Training loss is not quality.',
        'status_changed_during_collection': raw != small(volume, root + '/status.json')}


def launch(package, receipt, profile, authorize, resume_run=None):
    require(authorize, 'Explicit --authorize-gpu required')
    package, receipt = Path(package).resolve(), Path(receipt)
    value = verify_execution(package)
    require(not receipt.exists(), 'Submission exists; poll it, never resubmit')
    os.environ.update(MODAL_PROFILE=profile, GHA_R2_PACKAGE=str(package), GHA_R2_SOURCE_ROOT=str(package / 'frozen'))
    import modal
    if resume_run:
        require(value['mode'] == 'train', 'Smoke cannot resume main training')
        claim = modal.Dict.from_name('ghana-understanding-r2-invocations', create_if_missing=False)[run_id(resume_run)]
        outcome = modal.FunctionCall.from_id(claim['call_id']).get(timeout=0)
        terminal_pause(claim, outcome, resume_run, sha(package / 'execution.json'), value['data_sha256'])
    identifier = 'understanding_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    record = {'state': 'submitting', 'run_id': identifier, 'mode': value['mode'], 'profile': profile,
        'execution_sha256': sha(package / 'execution.json'), 'data_sha256': value['data_sha256'],
        'package': str(package), 'resume_run_id': resume_run, 'volume': VOLUME,
        'remote_directory': '/' + VERSION + '/' + identifier}
    save(receipt, record, exclusive=True)
    spec = importlib.util.spec_from_file_location('train_understanding_r2', package / 'r2/modal/train/train_understanding_r2.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with modal.enable_output(), module.app.run(detach=True):
        record['modal_image_id'] = module.image.object_id
        save(receipt, record)
        fn = module.smoke if value['mode'] == 'smoke' else module.train
        call = fn.spawn(identifier, record['execution_sha256'], resume_run)
        record.update(state='submitted', call_id=call.object_id)
        save(receipt, record)
    return record


def status(receipt, wait=0):
    value = read(receipt)
    require(value.get('call_id'), 'Submission outcome ambiguous; inspect claim, never automatically retry')
    os.environ['MODAL_PROFILE'] = value['profile']
    import modal
    try:
        result = modal.FunctionCall.from_id(value['call_id']).get(timeout=wait)
    except modal.exception.FunctionTimeoutError:
        result = {'state': 'failed', 'reason': 'modal_hard_timeout'}
    except TimeoutError:
        return {'state': 'pending', 'call_id': value['call_id']}
    save(Path(receipt).with_suffix('.result.json'), result)
    return result


def collect(receipt, out):
    value, out = read(receipt), Path(out)
    require(value['remote_directory'] == '/' + VERSION + '/' + run_id(value['run_id']), 'Receipt path mismatch')
    os.environ['MODAL_PROFILE'] = value['profile']
    import modal
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    collected = {}
    for name in ('execution.json', 'status.json', 'loading.json', 'smoke.json', 'stages.json', 'adapters.json', 'resume.json'):
        try:
            raw = b''.join(volume.read_file(value['remote_directory'] + '/' + name))
        except (FileNotFoundError, modal.exception.NotFoundError):
            continue
        require(len(raw) < 10_000_000, 'Metadata exceeds collection bound')
        parsed = json.loads(raw)
        if name == 'execution.json':
            require(hashlib.sha256(raw).hexdigest() == value['execution_sha256'], 'Remote execution mismatch')
        if name == 'status.json':
            require(parsed['run_id'] == value['run_id'] and parsed['execution_sha256'] == value['execution_sha256']
                    and parsed['data_sha256'] == value['data_sha256'], 'Remote status mismatch')
        out.mkdir(parents=True, exist_ok=True)
        if (out / name).exists():
            require((out / name).read_bytes() == raw, 'Collected evidence changed; use new output directory')
        else:
            with (out / name).open('xb') as handle:
                handle.write(raw)
        collected[name] = hashlib.sha256(raw).hexdigest()
    if value['mode'] == 'train':
        save(out / 'progress.json', progress(value, volume))
    return {'files': collected, 'directory': str(out)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'launch', 'status', 'collect'))
    p.add_argument('--pack', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--execution', type=Path); p.add_argument('--receipt', type=Path)
    p.add_argument('--mode', choices=('smoke', 'train'), default='smoke')
    p.add_argument('--evaluation', type=Path); p.add_argument('--smoke-directory', type=Path)
    p.add_argument('--resume-run'); p.add_argument('--profile', default='createdliving1000')
    p.add_argument('--authorize-gpu', action='store_true'); p.add_argument('--wait', type=int, default=0)
    a = p.parse_args()
    if a.action == 'prepare':
        result = prepare(a.pack, a.out, a.mode, a.evaluation, a.smoke_directory)
    elif a.action == 'launch':
        result = launch(a.execution, a.receipt, a.profile, a.authorize_gpu, a.resume_run)
    elif a.action == 'collect':
        result = collect(a.receipt, a.out)
    else:
        result = status(a.receipt, a.wait)
    print(json.dumps(result, indent=2))
