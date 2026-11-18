"""Private detached public benchmark; imports never define resources or start GPU work."""
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve()
CORE = HERE.parents[2] / 'scripts/r7_public_benchmark_protocol.py' if HERE.parent.name == 'train' else HERE.with_name('r7_public_benchmark_protocol.py')
sys.path.insert(0, str(CORE.parent))
spec = importlib.util.spec_from_file_location('_public_benchmark_core', CORE)
core = importlib.util.module_from_spec(spec)
exec(compile(CORE.read_bytes(), str(CORE), 'exec'), core.__dict__)

APP = 'ghana-r7-public-benchmark-private'
VOLUME = 'ghana-health-understand-train'
CACHE = 'ghana-health-understanding-hf-cache'
CLAIMS = 'ghana-r7-public-benchmark-invocations'
PREFIX = 'understanding-adaptation-r7/public-benchmark'


def prefix(tokenizer, prompt):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


class NativeRuntime:
    def __init__(self, bundle, binding):
        import torch
        manifest, module = core.bundle_module(bundle, binding['bundle_sha256'])
        self.runtime = module.Runtime(bundle, binding['bundle_sha256'])
        core.require(self.runtime.manifest == manifest, 'Loaded bundle differs')
        from native_adaptation_run_core import require_kernel_binding
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native_model
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        self.common = {'runtime': {p: importlib.metadata.version(p) for p in manifest['runtime']},
                       'tokenizer_files': {k: v for k, v in manifest['files'].items() if k.startswith('tokenizer/')},
                       'vocabulary_sha256': core.digest(self.runtime.tokenizer.get_vocab()),
                       'chat_template_sha256': core.digest(self.runtime.tokenizer.chat_template),
                       'key_mapping': {r'^model\.language_model\.': 'model.'},
                       'base_weights_verified': True, 'strict_loading': self.runtime.loading,
                       'no_meta_parameters': not any(p.is_meta for p in self.runtime.model.parameters()),
                       'dtype': str(self.runtime.model.dtype), 'device': self.runtime.model.device.type,
                       'attention': self.runtime.model.config._attn_implementation,
                       'tf32': torch.backends.cuda.matmul.allow_tf32,
                       'gpu_name': torch.cuda.get_device_name(), 'python': sys.version.split()[0],
                       'kernel_bindings': {
                           'chunk': require_kernel_binding(native_model.torch_chunk_gated_delta_rule, chunk_gated_delta_rule),
                           'recurrent': require_kernel_binding(native_model.torch_recurrent_gated_delta_rule,
                                                               fused_recurrent_gated_delta_rule)}}
        self.binding = binding

    def select(self, variant):
        import random
        import numpy as np
        import torch
        # select performs exact safetensors-to-live-adapter tensor equality checks.
        with self.runtime.select(variant):
            core.require(variant in self.runtime.loaded, 'Adapter was not verified')
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        proof = {'identity': self.runtime.identity(variant), 'common': self.common, 'adapter_values_exact': True}
        return core.verify_loading(proof, variant, self.binding)

    def score(self, row):
        import torch
        from transformers import GenerationConfig
        model, tokenizer = self.runtime.model, self.runtime.tokenizer
        text = prefix(tokenizer, row['prompt'])
        ids = tokenizer.encode(text, add_special_tokens=False)
        evidence = {'prompt_ids_sha256': core.digest(ids), 'prompt_tokens': len(ids)}
        if row['task'] == 'mcq':
            expected_ids, continuations = core.protocol.answer_tokens(tokenizer, text, row['labels'])
            core.require(ids == expected_ids, 'Tokenized prefix changed')

            def logprobs(sequence, targets, *, same_position):
                inputs = torch.tensor([sequence], device=model.device)
                keep = 1 if same_position else len(targets)
                with torch.inference_mode():
                    logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), logits_to_keep=keep).logits[0].float()
                    core.require(logits.shape[0] == keep, 'Unexpected causal-logit positions')
                    values = torch.log_softmax(logits, dim=-1)
                    return [values[0 if same_position else i, token].item() for i, token in enumerate(targets)]

            scores = core.conditional_scores(ids, continuations, logprobs)
            return {**evidence, 'prediction': core.protocol.choose(scores, row['labels']),
                    'scores': scores, 'answer_token_ids': continuations}
        inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to(model.device)
        core.require(inputs['input_ids'][0].tolist() == ids, 'SIB tokenized prefix differs')
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=64, repetition_penalty=1.0,
                                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=config)
        generated = output[0, len(ids):].tolist()
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {**evidence, 'prediction': core.protocol.classify_output(raw), 'raw_output': raw,
                'output_ids': generated, 'ended_with_eos': bool(generated and generated[-1] == tokenizer.eos_token_id)}


def verify_progress(progress, inputs, binding):
    core.require(progress['binding'] == binding and progress['snapshot_sha256'] == core.digest(
        {k: v for k, v in progress.items() if k != 'snapshot_sha256'}), 'Progress identity/hash differs')
    core.require(progress['state'] in ('loading', 'running', 'paused', 'error', 'complete'), 'Wrong progress state')
    rows = {r['id']: r for group in inputs.values() for r in group}
    proofs = progress['loadings']
    core.require(set(proofs) <= set(core.VARIANTS), 'Unknown loader variant')
    for variant, proof in proofs.items():
        core.verify_loading(proof, variant, binding)
    if len(proofs) == 2:
        core.require(proofs[core.VARIANTS[0]]['common'] == proofs[core.VARIANTS[1]]['common'], 'Paired loaders differ')
    for record in progress['records']:
        core.require(record['id'] in rows and record['variant'] in proofs, 'Unknown case or missing loader proof')
        core.verify_record(record, rows[record['id']], record['variant'], binding, proofs[record['variant']])
    measured = core.report(inputs, progress['records'])
    core.require(progress['report'] == measured, 'Progress report differs')
    core.require(progress['state'] != 'complete' or all(measured['total'][v]['pending'] == 0 for v in core.VARIANTS),
                 'Premature completion')
    return progress


def evaluate_local(root, inputs, binding, factory, *, attempt=0, call_id='cpu-test',
                   seconds=6900, commit=lambda: None, clock=time.monotonic):
    """Finalized cases are immutable; terminal attempts resume only missing cases."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    core.require(not root.is_symlink(), 'Linked result directory')
    identity = root / 'binding.json'
    if identity.exists():
        core.require(core.read(identity) == binding, 'Existing run belongs to another execution')
    else:
        core.require(not any(root.iterdir()), 'Unbound existing result directory')
        core.save(identity, binding)
        commit()
    tick = clock()
    progress_path = root / 'progress.json'
    progress = {'binding': binding, 'loadings': {}, 'records': [], 'state': 'loading', 'attempt': attempt,
                'call_id': call_id, 'error': None}
    if progress_path.exists():
        progress = verify_progress(core.read(progress_path), inputs, binding)
        core.require(attempt >= progress['attempt'], 'Stale attempt')
    was_complete = progress['state'] == 'complete'
    progress.update(attempt=attempt, call_id=call_id, state='loading', error=None)
    done = {(r['variant'], r['id']): r for r in progress['records']}
    rows = {r['id']: r for group in inputs.values() for r in group}
    paths = {(variant, row['id']): root / 'cases' / variant / name / f'{row["source_index"]:06d}.json'
             for variant in core.VARIANTS for name, group in inputs.items() for row in group}
    core.require(all(paths[key].is_file() for key in done), 'Finalized per-case artifact missing')
    # A process can die after the case write but before the progress checkpoint.
    for path in sorted((root / 'cases').rglob('*.json')):
        saved = core.read(path)
        record = saved['result']
        key = (record['variant'], record['id'])
        core.require(key in paths and path == paths[key] and not path.is_symlink(), 'Unexpected case artifact path')
        core.require(saved['input'] == rows[record['id']], 'Saved per-case input changed')
        core.verify_record(record, rows[record['id']], key[0], binding, progress['loadings'][key[0]])
        if key in done:
            core.require(done[key] == record, 'Finalized case changed')
        else:
            progress['records'].append(record)
            done[key] = record

    def checkpoint():
        progress['report'] = core.report(inputs, progress['records'])
        progress['snapshot_sha256'] = core.digest({k: v for k, v in progress.items() if k != 'snapshot_sha256'})
        core.save(progress_path, progress)
        commit()

    checkpoint()
    if was_complete:
        progress['state'] = 'complete'
        checkpoint()
        return progress
    try:
        runtime = factory()
        for variant in core.VARIANTS:
            if all((variant, key) in done for key in rows):
                continue
            proof = runtime.select(variant)
            core.verify_loading(proof, variant, binding)
            old = progress['loadings'].get(variant)
            core.require(old in (None, proof), 'Resumed loader proof differs')
            core.require(all(p['common'] == proof['common'] for p in progress['loadings'].values()), 'Paired runtime differs')
            progress['loadings'][variant] = proof
            progress['state'] = 'running'
            checkpoint()
            for name, group in inputs.items():
                for row in group:
                    key = (variant, row['id'])
                    if key in done:
                        continue
                    if clock() - tick >= seconds:
                        progress['state'] = 'paused'
                        checkpoint()
                        return progress
                    start = clock()
                    record = {'id': row['id'], 'input_sha256': row['input_sha256'], 'variant': variant,
                              'binding_sha256': core.digest(binding), 'loading_sha256': core.digest(proof),
                              'attempt': attempt, 'call_id': call_id, 'error': None}
                    try:
                        output = runtime.score(row)
                        record.update(state='scored', output=output, correct=output['prediction'] == row['gold'])
                        core.verify_record(record, row, variant, binding, proof)
                    except Exception as error:
                        record.update(state='error', output=None, correct=False,
                                      error={'type': type(error).__name__, 'message': str(error)[:2000]})
                    record['seconds'] = clock() - start
                    path = root / 'cases' / variant / name / f'{row["source_index"]:06d}.json'
                    core.require(not path.exists(), 'Never overwrite finalized evidence')
                    core.save(path, {'input': row, 'result': record})
                    progress['records'].append(record)
                    done[key] = record
                    if len(done) % 25 == 0 or record['state'] == 'error':
                        checkpoint()
                    if record['state'] == 'error':
                        raise RuntimeError('Case failed; resume remaining cases. Failed case remains in denominator: ' + row['id'])
            checkpoint()
        progress['state'] = 'complete'
        checkpoint()
    except Exception as error:
        progress.update(state='error', error={'type': type(error).__name__, 'message': str(error)[:2000]})
        checkpoint()
        raise
    return progress


def terminal_state(graph):
    core.require(graph, 'No terminal call evidence; do not resubmit')
    states = []
    for node in graph:
        state = node.status.name
        core.require(state in {'SUCCESS', 'FAILURE', 'INIT_FAILURE', 'TERMINATED', 'TIMEOUT'},
                     'Previous call may still be active; do not resubmit')
        states.append(state)
        if node.children:
            states.append(terminal_state(node.children))
    return 'SUCCESS' if all(s == 'SUCCESS' for s in states) else 'FAILED'


def claim_key(run_id, attempt):
    core.validate_run_id(run_id)
    core.require(type(attempt) is int and 0 <= attempt <= 100, 'Invalid attempt')
    return run_id + ':attempt-' + str(attempt)


def require_launch_python():
    core.require(sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 11),
                 'Launch requires CPython 3.11 matching the pinned image')


def remote_evaluate(expected_sha, attempt, volume):
    import modal
    execution, inputs = core.verify_execution('/public', expected_sha)
    binding = execution['binding']
    run_id = binding['run_id']
    claims = modal.Dict.from_name(CLAIMS, create_if_missing=True)
    owner = modal.current_function_call_id()
    if attempt:
        previous = claims.get(claim_key(run_id, attempt - 1))
        core.require(previous is not None and previous['execution_sha256'] == expected_sha, 'Missing previous attempt')
        terminal_state(modal.FunctionCall.from_id(previous['call_id']).get_call_graph())
    claim = {'call_id': owner, 'execution_sha256': expected_sha, 'attempt': attempt}
    key = claim_key(run_id, attempt)
    claims.put(key, claim, skip_if_exists=True)
    core.require(claims.get(key) == claim, 'GPU attempt is already owned')
    volume.reload()
    return evaluate_local(Path('/artifacts') / PREFIX / run_id, inputs, binding,
                          lambda: NativeRuntime('/bundle', binding), attempt=attempt, call_id=owner, commit=volume.commit)


def create_app(package, expected_sha, *, modal_module=None):
    require_launch_python()
    execution, _ = core.verify_execution(package, expected_sha)
    manifest, _ = core.bundle_module(execution['bundle'], execution['binding']['bundle_sha256'])
    if modal_module is None:
        import modal as modal_module
    modal = modal_module
    app = modal.App(APP)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    cache = modal.Volume.from_name(CACHE, create_if_missing=False)
    image = modal.Image.debian_slim(python_version='3.11').pip_install(
        *[name + '==' + version for name, version in manifest['runtime'].items()]).env({
            'PYTHONDONTWRITEBYTECODE': '1', 'HF_HOME': '/cache/hf', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'DO_NOT_TRACK': '1',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    image = image.add_local_dir(str(Path(package).resolve()), '/public')
    image = image.add_local_dir(str(Path(execution['bundle']).resolve()), '/bundle')

    @app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=7200, retries=0,
                  max_containers=1, serialized=True, volumes={'/cache': cache.read_only(), '/artifacts': volume})
    def evaluate(attempt):
        import hashlib
        import importlib.util
        import json
        from pathlib import Path
        root = Path('/public')
        raw = (root / 'execution.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError('Remote execution identity changed')
        for name, checksum in json.loads(raw)['files'].items():
            if name.startswith('code/') and hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Remote executable changed')
        path = root / 'code/worker.py'
        spec = importlib.util.spec_from_file_location('_verified_public_worker', path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        return module.remote_evaluate(expected_sha, attempt, volume)

    return app, evaluate
