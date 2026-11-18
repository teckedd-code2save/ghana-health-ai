"""Future R7 inference contract. No R7 checkpoint or completion is assumed."""
import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import re

_path = Path(__file__).with_name('r2_inference.py')
_source = _path.read_bytes()
if hashlib.sha256(_source).hexdigest() != 'c8ac9598d3108e4d93774a0dfe1cb753f5793460cc81a5fec73a5b75dac075f2':
    raise ValueError('Pinned R2 runtime differs before import')
_spec = importlib.util.spec_from_file_location('_r7_pinned_r2_runtime', _path)
base = importlib.util.module_from_spec(_spec)
exec(compile(_source, str(_path), 'exec'), base.__dict__)
read, sha, safe, require, digest = base.read, base.sha, base.safe, base.require, base.digest
validate_messages = base.validate_messages
VERSION = 'r7-comparison-inference-v1'
VARIANTS = ('base', 'r6_response', 'r7_response')
R6_BUNDLE_SHA = '529200bb9482784bf7045985de3ffda7a712055978ec9f302f19dfeec4ed670a'
R6_DIRECTORY = '/artifacts/understanding-adaptation-r6/understanding_20261113T152209Z/responses-adapter'
R6_WEIGHTS = '6e56d8c7c04f456f94579d4bd82064521c97d0b9e23dbbef1e9783280568861c'
R6_CONFIG = '577b80656dbcff84886c4e402022d9507abbef34b0e0013bceab029ccf4c9403'
INITIALIZATION = {'adapter_directory': R6_DIRECTORY, 'adapter_weights_sha256': R6_WEIGHTS,
                  'adapter_config_sha256': R6_CONFIG, 'new_optimizer': True, 'new_scheduler': True}
APP_NAME = 'ghana-understanding-r7-demo-private'
CLASS_NAME = 'UnderstandingR7Demo'
VOLUME = 'ghana-health-understand-train'
HELPER_HASHES = {
    'understanding_r2_demo.py': 'e11649ff095b6315a06b02ac72661e4e4006aefb73c9eceb8e851539ce127781',
    'guarded_r3_runtime.py': '78b67c116cf6edd9b43636c52519ee294847fe41db0077feb8101aef9e3bfb92',
    'response_loop_guard.py': '2b0830e33dd3a9bea182d2dbbd4d9d8c091f14eaea4b23e95a6664391ac3b474',
}


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def check_parent(parent, checksum):
    require(checksum == R6_BUNDLE_SHA and parent['version'] == 'r6-comparison-inference-v1',
            'Use the unchanged pinned R6 bundle')
    require(parent['model'] == base.MODEL and parent['revision'] == base.REVISION
            and parent['base_sha256'] == base.BASE_SHA and parent['runtime'] == base.RUNTIME
            and parent['decoding'] == base.DECODING and parent['variants']['base'] is None
            and parent['variants']['r6_response'] == {'directory': R6_DIRECTORY, 'files': {
                'adapter_config.json': R6_CONFIG, 'adapter_model.safetensors': R6_WEIGHTS}}, 'R6 protocol/adapter differs')


def receipt_root(receipt):
    identifier = receipt.get('run_id')
    require(isinstance(identifier, str) and re.fullmatch(r'understanding_[0-9]{8}T[0-9]{6}Z', identifier),
            'A real R7 training run ID is required; placeholders are not checkpoints')
    root = '/artifacts/understanding-adaptation-r7/' + identifier
    require(receipt.get('state') == 'submitted' and receipt.get('mode') == 'train'
            and receipt.get('volume') == VOLUME and receipt.get('remote_directory') == root.removeprefix('/artifacts')
            and isinstance(receipt.get('call_id'), str) and receipt['call_id'].startswith('fc-')
            and is_sha(receipt.get('execution_sha256')) and is_sha(receipt.get('data_sha256')),
            'Submitted, hash-bound R7 train receipt required; smoke is not a checkpoint')
    return root


def training_proof(receipt, terminal, fetch):
    """Read actual completion/export commitments; no placeholder adapter hashes."""
    root = receipt_root(receipt)
    identifier = receipt['run_id']
    status, _ = fetch(root + '/status.json')
    execution, execution_sha = fetch(root + '/execution.json')
    data, data_sha = fetch(root + '/inputs/manifest.json')
    loading, _ = fetch(root + '/loading.json')
    exports, _ = fetch(root + '/adapters.json')
    require(status == terminal and status.get('state') == 'completed' and status.get('training_completed') is True
            and status.get('production_promoted') is False and status['run_id'] == identifier
            and status['mode'] == execution['mode'] == 'train', 'Completed R7 training is required')
    require(execution_sha == receipt['execution_sha256'] == status['execution_sha256']
            and data_sha == receipt['data_sha256'] == status['data_sha256'] == execution['data_sha256'],
            'R7 execution/data hashes differ')
    require(status['remote_directory'] == root and execution['version'] == data['version'] == 'understanding-adaptation-r7'
            and execution['runtime'] == loading['runtime'] == base.RUNTIME and loading['weight_sha256'] == base.BASE_SHA
            and data['model'] == base.MODEL and data['revision'] == base.REVISION
            and execution['private_research_only'] is True and data['private_research_only'] is True
            and data['production_eligible'] is False, 'Wrong R7 foundation/runtime/research scope')
    require(all(record['initialization'] == INITIALIZATION for record in (status, execution, data, loading))
            and loading['initial_adapter_values_exact'] is True and loading['frozen_base'] is True
            and loading['r6_optimizer_loaded'] is False and type(loading['r7_full_state_resume']) is bool
            and loading['new_vocabulary'] is False and loading['trainable_parameters'] == 43278336,
            'R7 must continue exact R6 adapter values with a frozen foundation')
    smoke = execution.get('smoke', {})
    require(all(is_sha(smoke.get(key)) for key in ('execution_sha256', 'status_sha256', 'smoke_sha256')),
            'Missing completed training smoke binding')
    require(isinstance(smoke.get('run_id'), str)
            and re.fullmatch(r'understanding_[0-9]{8}T[0-9]{6}Z', smoke['run_id']) and smoke['run_id'] != identifier,
            'A separate completed smoke run is required')
    smoke_root = '/artifacts/understanding-adaptation-r7/' + smoke['run_id']
    smoke_status, smoke_status_sha = fetch(smoke_root + '/status.json')
    smoke_execution, smoke_execution_sha = fetch(smoke_root + '/execution.json')
    smoke_result, smoke_result_sha = fetch(smoke_root + '/smoke.json')
    require(smoke_status_sha == smoke['status_sha256'] and smoke_execution_sha == smoke['execution_sha256']
            and smoke_result_sha == smoke['smoke_sha256'] and smoke_status['state'] == 'completed'
            and smoke_status['smoke_passed'] is True and smoke_status['mode'] == smoke_execution['mode'] == 'smoke'
            and smoke_status['run_id'] == smoke['run_id'] and smoke_status['remote_directory'] == smoke_root
            and smoke_status['execution_sha256'] == smoke_execution_sha
            and smoke_status['data_sha256'] == smoke_execution['data_sha256'] == data_sha
            and smoke_execution['version'] == 'understanding-adaptation-r7'
            and smoke_execution['runtime'] == base.RUNTIME and smoke_execution['initialization'] == INITIALIZATION
            and smoke_result['passed'] is True and smoke_result['real_weight_backward'] is True
            and smoke_result['initial_adapter_values_exact'] is True
            and smoke_result['native_encoding']['all_rows_verified'] > 0,
            'Completed R7 smoke evidence does not match this training')
    require(exports == status['adapters'] and len(status['stages']) == len(execution['expected_stages']) == 1,
            'Incomplete R7 final stage/export evidence')
    actual, planned = status['stages'][0], execution['expected_stages'][0]
    position = actual['position']
    require(actual['stage'] == planned['name'] == 'responses' and actual['status'] == 'completed'
            and actual['completion_verified'] is True and type(planned['optimizer_updates']) is int
            and planned['optimizer_updates'] > 0 and position['completed_updates'] == planned['optimizer_updates']
            and position['completed_microbatches'] == planned['scheduled_sequences']
            and position['microbatches_in_update'] == 0
            and position['exposure']['effective_target_tokens'] == planned['effective_target_tokens']
            and position['exposure']['input_tokens'] == planned['sequence_tokens'], 'Incomplete final training exposure')
    physical = PurePosixPath(status['physical_directory'])
    require(physical.is_absolute() and '..' not in physical.parts
            and physical.parts[-2:] == ('understanding-adaptation-r7', identifier), 'Invalid physical R7 directory')

    def remote(path):
        path = PurePosixPath(path)
        require('..' not in path.parts and path.is_relative_to(physical), 'Evidence outside final R7 run')
        return root + '/' + path.relative_to(physical).as_posix()

    directory = root + '/responses-adapter'
    # A resumed run retains earlier pass snapshots at their original locations.
    finals = [export for export in exports if PurePosixPath(export['directory']).name == 'responses-adapter']
    require(len(finals) == 1, 'Exactly one final R7 adapter required; pass snapshots are not final')
    export = finals[0]
    updates = position['completed_updates']
    require(remote(export['directory']) == directory and export['stage'] == 'responses'
            and export['checkpoint_adapter_tensors_verified'] is True
            and export['completed_updates'] == updates and export['initialization'] == INITIALIZATION
            and {'adapter_config.json', 'adapter_model.safetensors'} <= set(export['files'])
            <= {'adapter_config.json', 'adapter_model.safetensors', 'README.md'}
            and all(is_sha(value) for value in export['files'].values())
            and export['files']['adapter_model.safetensors'] != R6_WEIGHTS, 'Unverified/unchanged final R7 export')
    config, config_sha = fetch(directory + '/adapter_config.json')
    base.adapter_config(config)
    require(config_sha == export['files']['adapter_config.json'], 'R7 adapter configuration readback differs')
    checkpoint = export['checkpoint']
    remote_checkpoint = remote(checkpoint['directory'])
    require(PurePosixPath(remote_checkpoint).name == 'update-%08d' % updates
            and actual['checkpoint'] == checkpoint['directory'], 'Final R7 checkpoint differs')
    cp_manifest, cp_sha = fetch(remote_checkpoint + '/manifest.json')
    committed, _ = fetch(remote_checkpoint + '/COMMITTED.json')
    contract, contract_sha = fetch(str(PurePosixPath(remote_checkpoint).parent.parent) + '/contract.json')
    require(cp_sha == checkpoint['manifest_sha256'] and committed == {'manifest_sha256': cp_sha}
            and is_sha(checkpoint['state_sha256']) and cp_manifest['state_sha256'] == checkpoint['state_sha256']
            and contract_sha == checkpoint['contract_sha256'] and contract['run_sha256'] == execution_sha
            and contract['data_sha256'] == data_sha
            and cp_manifest['identity'] == {'run_sha256': digest(contract), 'data_sha256': data_sha},
            'R7 checkpoint commitment differs')
    consumption, consumption_sha = fetch(root + '/consumption.json')
    require(consumption_sha == status['consumption_sha256'] and consumption['version'] == 'r7-consumption-v1'
            and consumption['run_id'] == identifier and consumption['mode'] == 'train'
            and consumption['training_complete'] is True and consumption['schedule_complete'] is True
            and consumption['execution_sha256'] == execution_sha and consumption['data_sha256'] == data_sha
            and consumption['checkpoint'] == checkpoint and consumption['completed_updates'] == updates
            and consumption['consumed']['row_exposures'] == consumption['planned']['row_exposures'] == planned['scheduled_sequences']
            and consumption['consumed']['effective_target_tokens'] == consumption['planned']['effective_target_tokens'] == planned['effective_target_tokens']
            and consumption['consumed']['input_tokens'] == consumption['planned']['input_tokens'] == planned['sequence_tokens'],
            'R7 committed consumption differs')
    return {'run_id': identifier, 'call_id': receipt['call_id'], 'execution_sha256': execution_sha,
            'data_sha256': data_sha, 'training_completed': True, 'consumption_sha256': consumption_sha,
            'checkpoint': checkpoint, 'completed_updates': updates, 'variant': {'directory': directory, 'files': {
                key: export['files'][key] for key in ('adapter_config.json', 'adapter_model.safetensors')}}}


def verify_files(folder, expected_sha):
    folder = Path(folder).resolve()
    require(is_sha(expected_sha) and sha(folder / 'manifest.json') == expected_sha, 'Bundle identity changed')
    manifest = read(folder / 'manifest.json')
    require(manifest['version'] == VERSION, 'Completed R7 bundle required')
    require(not any(path.is_symlink() for path in folder.rglob('*')), 'Linked R7 bundle artifact')
    require({path.relative_to(folder).as_posix() for path in folder.rglob('*') if path.is_file()}
            - {'manifest.json', 'SEALED.json'} == set(manifest['files']), 'Unbound or missing R7 artifact')
    for name, checksum in manifest['files'].items():
        require(is_sha(checksum) and sha(safe(folder, name)) == checksum, 'R7 bundle artifact changed: ' + name)
    if (folder / 'SEALED.json').exists():
        require(read(folder / 'SEALED.json') == {'manifest_sha256': expected_sha}, 'R7 bundle seal differs')
    return manifest


def load_bundle(folder, expected_sha=None):
    folder = Path(folder)
    manifest = verify_files(folder, expected_sha or sha(folder / 'manifest.json'))
    require(manifest['model'] == base.MODEL and manifest['revision'] == base.REVISION
            and manifest['base_sha256'] == base.BASE_SHA and manifest['runtime'] == base.RUNTIME
            and manifest['decoding'] == base.DECODING and manifest['private'] is True and manifest['training'] is False
            and manifest['default_variant'] == 'r7_response' and set(manifest['variants']) == set(VARIANTS)
            and manifest['variants']['base'] is None, 'R7 inference protocol/checkpoints changed')
    required = {'code/r2_inference.py', 'code/r7_inference.py', 'code/research_understanding_r7_demo.py',
                'r6-manifest.json', 'evidence/receipt.json', 'evidence/terminal.json', 'evidence/adapter-readback.json'}
    require(required | {'code/' + name for name in HELPER_HASHES} <= set(manifest['files']), 'Missing R7 bundle evidence')
    parent = read(folder / 'r6-manifest.json')
    check_parent(parent, sha(folder / 'r6-manifest.json'))
    require(manifest['r6_bundle_sha256'] == R6_BUNDLE_SHA
            and manifest['variants']['r6_response'] == parent['variants']['r6_response'], 'R6 parent binding changed')
    for name, checksum in parent['files'].items():
        if name.startswith(('tokenizer/', 'code/')):
            require(manifest['files'].get(name) == checksum, 'Inherited tokenizer/runtime changed')
    require(all(manifest['files']['code/' + name] == checksum for name, checksum in HELPER_HASHES.items()),
            'Pinned streaming/guard helper changed')
    remote_files = {}

    def fetch(remote):
        require(remote.startswith('/artifacts/understanding-adaptation-r7/'), 'Readback outside R7')
        name = 'evidence/readback/' + remote.removeprefix('/artifacts/')
        require(name in manifest['files'], 'Unbound R7 training evidence')
        path = safe(folder, name)
        remote_files[remote] = sha(path)
        return read(path), sha(path)

    require(sha(folder / 'evidence/receipt.json') == manifest['r7_receipt_sha256'], 'R7 receipt binding differs')
    proof = training_proof(read(folder / 'evidence/receipt.json'), read(folder / 'evidence/terminal.json'), fetch)
    require(manifest['r7_training'] == proof and manifest['variants']['r7_response'] == proof['variant']
            and manifest['remote_files'] == remote_files, 'R7 proof/variant/evidence differs')
    readbacks = read(folder / 'evidence/adapter-readback.json')
    require(set(readbacks) == {'r6_response', 'r7_response'}, 'Missing adapter readback proof')
    for variant, result in readbacks.items():
        adapter = manifest['variants'][variant]
        require(result['files'] == adapter['files'] and result['directory'] == adapter['directory']
                and result['verified'] is True and all(type(result['bytes_read'][name]) is int
                    and result['bytes_read'][name] > 0 for name in adapter['files']), 'Actual adapter readback differs')
    require(manifest['actor'] == {'app_name': APP_NAME, 'class_name': CLASS_NAME}, 'Use the separate private R7 actor')
    return manifest


def bound_bundle(folder, expected_sha):
    verify_files(folder, expected_sha)
    source = safe(Path(folder), 'code/r7_inference.py')
    spec = importlib.util.spec_from_file_location('_r7_bundle_' + expected_sha, source)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
    return module.load_bundle(folder, expected_sha), module


class Runtime(base.Runtime):
    def __init__(self, bundle_dir, expected_sha, artifacts_root=Path('/artifacts')):
        super().__init__(bundle_dir, expected_sha, artifacts_root, bundle_loader=load_bundle)
