"""One bounded private R2 worker, using byte-pinned original numerical helpers."""
import os
from pathlib import Path
import sys
import modal

PACKAGE = Path(os.environ['GHA_R2_PACKAGE']).resolve() if modal.is_local() else Path('/r2-package')
sys.path.insert(0, str(PACKAGE / 'r2/scripts'))
from run_understanding_r2 import VERSION, RUNTIME, verify_execution

CONTRACT = verify_execution(PACKAGE)
app = modal.App('ghana-understanding-adaptation-r2')
artifacts = modal.Volume.from_name('ghana-health-understand-train', create_if_missing=False)
cache = modal.Volume.from_name('ghana-health-understanding-hf-cache', create_if_missing=False)
claims = modal.Dict.from_name('ghana-understanding-r2-invocations', create_if_missing=True)
image = modal.Image.debian_slim(python_version='3.11').pip_install(
    *[name + '==' + version for name, version in RUNTIME.items()]
).env({'HF_HOME': '/cache/hf', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
       'TOKENIZERS_PARALLELISM': 'false', 'DO_NOT_TRACK': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
       'GHA_R2_SOURCE_ROOT': '/r2-package/frozen',
       'PYTHONPATH': '/r2-package/r2/modal/train:/r2-package/r2/scripts:/r2-package/frozen/scripts:/r2-package/frozen/modal/train'})
if modal.is_local():
    image = image.add_local_dir(str(PACKAGE), '/r2-package')


def execute(identifier, execution_sha, mode, resume_run=None):
    import importlib.metadata
    import json
    import random
    import shutil
    import time
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native_model
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from run_understanding_r2 import (activate, pack_data, require, read, sha, save, run_id, mounted_root,
        terminal_pause, inherit_prefix, publish_adapter, check_stage, verify_adapter_tensors)

    started = time.monotonic()
    run_id(identifier)
    require(sha(PACKAGE / 'execution.json') == execution_sha, 'Execution invocation changed')
    contract = verify_execution(PACKAGE)
    require(mode == contract['mode'], 'Wrong execution mode')
    require({n: importlib.metadata.version(n) for n in RUNTIME} == RUNTIME, 'Installed runtime differs')
    activate(PACKAGE / 'frozen')
    from afrique_instruction_baseline_core import WEIGHT, verify_weight, verify_loading, verify_cache_receipt
    from check_afrique_training_contract import ASSET_HASHES, MODEL, REVISION, load_tokenizer, encode_complete
    from prepare_understanding_r2 import TARGET_MODULES, encode_language
    from native_adaptation_run_core import require_kernel_binding
    from response_resume_state import trainable_state_dict
    from understanding_training_core import train_schedule, assert_checkpoint_equivalent
    from understanding_recovery import select_recovery
    from adapter_checkpoint import save_adapter

    manifest, rows, stages = pack_data(PACKAGE / 'inputs', PACKAGE / 'frozen')
    require(claims.put(identifier, {'call_id': modal.current_function_call_id(), 'execution_sha256': execution_sha},
                       skip_if_exists=True), 'Run already claimed; never resubmit')
    artifacts.reload(); cache.reload()
    root = mounted_root('/artifacts')
    destination = root / identifier
    destination.mkdir(exist_ok=False)
    results, adapter_records, selection = [], [], None

    def record(name, value):
        save(destination / name, value)
        artifacts.commit()
        return value

    def status(state, **extra):
        return record('status.json', {'state': state, 'mode': mode, 'run_id': identifier,
            'execution_sha256': execution_sha, 'data_sha256': contract['data_sha256'],
            'evaluation_manifest_sha256': contract['evaluation']['sha256'],
            'remote_directory': '/artifacts/' + VERSION + '/' + identifier,
            'physical_directory': str(destination), 'elapsed_seconds': time.monotonic() - started,
            'training_completed': False, 'production_promoted': False, **extra})

    try:
        shutil.copyfile(PACKAGE / 'execution.json', destination / 'execution.json')
        shutil.copytree(PACKAGE / 'inputs', destination / 'inputs')
        shutil.copytree(PACKAGE / 'evaluation', destination / 'evaluation')
        shutil.copytree(PACKAGE / 'frozen', destination / 'frozen')
        shutil.copytree(PACKAGE / 'r2', destination / 'r2')
        record('stages.json', results)
        record('adapters.json', adapter_records)
        status('loading')
        if mode == 'train':
            proof = contract['smoke']
            smoke_run = root / run_id(proof['run_id'])
            require(not smoke_run.is_symlink() and smoke_run.resolve() == smoke_run, 'Unsafe smoke directory')
            for name, field in (('execution.json', 'execution_sha256'), ('smoke.json', 'smoke_sha256'), ('status.json', 'status_sha256')):
                require(sha(smoke_run / name) == proof[field], 'Remote real-weight smoke proof changed')
            require(read(smoke_run / 'status.json')['state'] == 'completed' and read(smoke_run / 'smoke.json')['passed'] is True,
                    'Remote smoke did not complete')
        if resume_run:
            require(mode == 'train', 'Only main training resumes')
            previous = root / run_id(resume_run)
            require(previous != destination and previous.resolve() == previous, 'Unsafe recovery directory')
            claim = claims[resume_run]
            outcome = modal.FunctionCall.from_id(claim['call_id']).get(timeout=0)
            terminal = terminal_pause(claim, outcome, resume_run, execution_sha, contract['data_sha256'])
            require(outcome == read(previous / 'status.json'), 'Previous pause status changed')
            selection = select_recovery(previous, runs_root=root, execution_sha256=execution_sha,
                data_sha256=contract['data_sha256'], stages=stages,
                grad_accum=manifest['optimization']['grad_accum'], claim=claim, terminal_call=terminal)
            results, adapter_records, selection = inherit_prefix(previous, destination, stages, selection, contract)
            record('stages.json', results)
            record('adapters.json', adapter_records)
            record('resume.json', {**selection, 'previous_run_id': resume_run, 'terminal_call': terminal})

        snapshot = Path('/cache/hf/hub', 'models--' + MODEL.replace('/', '--'), 'snapshots', REVISION)
        verify_cache_receipt(read(snapshot / 'AFRIQUE_CACHE_READY.json'))
        for name, checksum in ASSET_HASHES.items():
            require(sha(snapshot / name) == checksum, 'Cached model asset changed')
        weight_sha = verify_weight(snapshot / 'model.safetensors', WEIGHT)
        with safe_open(snapshot / 'model.safetensors', framework='pt', device='cpu') as handle:
            shapes = {name: list(handle.get_slice(name).get_shape()) for name in handle.keys()}
        kernels = {'chunk': require_kernel_binding(native_model.torch_chunk_gated_delta_rule, chunk_gated_delta_rule),
                   'recurrent': require_kernel_binding(native_model.torch_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule)}
        base, info = Qwen3_5ForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa',
            key_mapping={r'^model\.language_model\.': 'model.'}, output_loading_info=True)
        loading = verify_loading(info, shapes, {k: list(v.shape) for k, v in base.state_dict().items()})
        require(not any(p.is_meta for p in base.parameters()), 'Unmaterialized base weights')
        # Language seed 40 controls only the builder's order; initialization remains R1 seed 42.
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        base.requires_grad_(False)
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
            target_modules=TARGET_MODULES, bias='none', task_type='CAUSAL_LM'))
        model.peft_config['default'].base_model_name_or_path = MODEL
        model.peft_config['default'].revision = REVISION
        model.enable_input_require_grads(); model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        trainables = {n: p for n, p in model.named_parameters() if p.requires_grad}
        require(trainables and all('lora_' in n for n in trainables) and
                sum(p.numel() for p in trainables.values()) == 43278336 and
                all(p.dtype == torch.float32 for p in trainables.values()), 'Unexpected adapter parameters')
        record('loading.json', {**loading, 'weight_sha256': weight_sha, 'kernels': kernels, 'runtime': RUNTIME,
            'trainable_parameters': 43278336, 'frozen_base': True, 'fresh_lora_seed': 42,
            'gpu': torch.cuda.get_device_name(), 'new_vocabulary': False})
        if adapter_records:
            verify_adapter_tensors(model, adapter_records)
            record('adapters.json', adapter_records)
        if mode == 'smoke':
            indices = [max((i for i, r in enumerate(rows) if r['collection'] == name),
                           key=lambda i: rows[i]['sequence_tokens'])
                       for name in ('language', 'meaning', 'responses', 'english_retention')]
            tokenizer = load_tokenizer(PACKAGE / 'inputs/tokenizer')
            witnesses = [json.loads(line) for line in (PACKAGE / 'inputs/source-witnesses.jsonl').read_text().splitlines()]
            for i in indices:
                witness = witnesses[i]
                encoded = encode_language(tokenizer, witness['text']) if witness['collection'] == 'language' else encode_complete(tokenizer, witness, 4096)
                require(all(encoded[k] == rows[i][k] for k in ('input_ids', 'attention_mask', 'labels')), 'Native smoke encoding differs')
            initial = trainable_state_dict(model)
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            options = {'run_sha256': execution_sha, 'data_sha256': contract['data_sha256'], 'learning_rate': 5e-5,
                       'grad_accum': 2, 'checkpoint_every': 1, 'commit': artifacts.commit}
            status('real_weight_backward_and_recovery')
            continuous = train_schedule(model, rows, indices, output=destination / 'uninterrupted', **options)
            expected = trainable_state_dict(model)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in initial['parameters']:
                        p.copy_(initial['parameters'][name])
                for name, b in model.named_buffers():
                    b.copy_(initial['buffers'][name])
            model.zero_grad(set_to_none=True)
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
            interrupted = train_schedule(model, rows, indices, output=destination / 'interrupted', stop_after_updates=1, **options)
            resumed = train_schedule(model, rows, indices, output=destination / 'resumed', resume=Path(interrupted['checkpoint']), **options)
            actual = trainable_state_dict(model)
            equal = all(torch.equal(expected[k][n], actual[k][n]) for k in expected for n in expected[k])
            require(equal, 'Real-weight resume did not match bitwise')
            recovery = assert_checkpoint_equivalent(continuous['checkpoint'], resumed['checkpoint'])
            record('smoke.json', {'passed': True, 'real_weight_backward': True, 'finite_gradients_checked_by_core': True,
                'resume_parameter_and_buffer_bitwise_equal': equal, 'full_state_recovery': recovery,
                'native_encoding_kinds': ['language', 'chat'], 'probe_rows': [rows[i]['id'] for i in indices],
                'uninterrupted': continuous, 'interrupted': interrupted, 'resumed': resumed,
                'not_a_research_adapter': True})
            return status('completed', smoke_passed=True)

        start_stage = selection['stage_index'] if selection else 0
        for index, stage in enumerate(stages):
            if index < start_stage:
                continue
            remaining = contract['soft_seconds'] - (time.monotonic() - started)
            status('training', stage=stage['name'])
            result = train_schedule(model, rows, stage['schedule'], output=destination / stage['name'],
                run_sha256=execution_sha, data_sha256=contract['data_sha256'],
                resume=Path(selection['checkpoint']) if selection and index == start_stage else None,
                max_seconds=max(0, remaining - 60), commit=artifacts.commit,
                **{k: manifest['optimization'][k] for k in ('learning_rate', 'grad_accum', 'warmup_fraction', 'checkpoint_every')})
            results.append({'stage': stage['name'], **result})
            record('stages.json', results)
            if selection and index == start_stage:
                restored = Path(result['attempt']) / ('update-%08d' % selection['completed_updates'])
                record('restore-proof.json', assert_checkpoint_equivalent(selection['checkpoint'], restored))
            if result['status'] != 'completed':
                return status('paused_budget', stages=results, automatic_restart=False)
            check_stage(results[-1], stage)
            target = destination / (stage['name'] + '-adapter')
            save_adapter(model, target)
            adapter_records.append(publish_adapter(target, result['checkpoint'], execution_sha, contract['data_sha256']))
            verify_adapter_tensors(model, adapter_records[-1:])
            record('adapters.json', adapter_records)
        require(len(results) == 3 and len(adapter_records) == 3, 'Incomplete stage exports')
        save_adapter(model, destination / 'adapter')
        artifacts.commit()
        return status('completed', stages=results, adapters=adapter_records, training_completed=True)
    except Exception as error:
        status('failed', stages=results, error_type=type(error).__name__, error=str(error))
        raise


@app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=900, retries=0,
              min_containers=0, max_containers=1, single_use_containers=True,
              volumes={'/cache': cache.read_only(), '/artifacts': artifacts}, include_source=False)
def smoke(identifier: str, execution_sha: str, resume_run: str | None = None):
    return execute(identifier, execution_sha, 'smoke', resume_run)


@app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=14400, retries=0,
              min_containers=0, max_containers=1, single_use_containers=True,
              volumes={'/cache': cache.read_only(), '/artifacts': artifacts}, include_source=False)
def train(identifier: str, execution_sha: str, resume_run: str | None = None):
    return execute(identifier, execution_sha, 'train', resume_run)
