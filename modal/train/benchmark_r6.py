"""Versioned, resumable, matched R6/R5/base benchmarks. Never overwrite v1 results."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

import modal

LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[2] if LOCAL else Path('/code')
sys.path.insert(0, str(ROOT / 'scripts'))
import r6_benchmark_protocol as protocol

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers==5.17.0", "peft==0.20.0", "torch==2.13.0", "accelerate==1.15.0",
    "datasets==5.0.0", "sentencepiece",
).env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"})
if LOCAL:
    image = image.add_local_file(str(ROOT / 'scripts/r6_benchmark_protocol.py'), '/code/scripts/r6_benchmark_protocol.py')
    image = image.add_local_file(__file__, '/code/modal/train/benchmark_r6.py')

app = modal.App("ghana-health-r6-benchmark-v2")
train_vol = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
cache_vol = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
VOLUMES = {"/cache": cache_vol, "/artifacts": train_vol}
MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REVISION = "693f12e444a30971ca53889128d7bade9491ecb3"
ADAPTERS = {
    "r5": "/artifacts/understanding-adaptation-r5/understanding_20261110T103734Z/responses-adapter",
    "r6": "/artifacts/understanding-adaptation-r6/understanding_20261113T152209Z/responses-adapter",
}
RUN_ROOT = Path('/artifacts/understanding-adaptation-r6/benchmark-corrections')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_path(run_id):
    protocol.require(re.fullmatch(r'[a-zA-Z0-9_-]{8,80}', run_id), 'Invalid run ID')
    return RUN_ROOT / run_id


def code_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT / 'scripts/r6_benchmark_protocol.py', ROOT / 'modal/train/benchmark_r6.py')}


@app.function(image=image, volumes=VOLUMES, timeout=1800, retries=0)
def prepare(run_id: str, limit: int = 300):
    from collections import Counter
    from datasets import load_dataset
    from huggingface_hub import HfApi
    protocol.require(type(limit) is int and limit > 0, 'Explicit positive sample limit required')
    root = run_path(run_id)
    protocol.require(not root.exists(), 'Run directory exists; use resume, never overwrite')
    root.mkdir(parents=True)
    old = Path('/artifacts/understanding-adaptation-r6/benchmark-results')
    legacy = {}
    for variant in protocol.VARIANTS:
        raw = (old / f'{variant}.json').read_bytes()
        dest = root / 'legacy' / f'{variant}.json'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(raw)
        legacy[variant] = {'sha256': sha(dest), 'original_path': str(old / dest.name)}
    snapshots, sources = {}, {}
    for name, (repo, config, split) in protocol.SOURCES.items():
        revision = HfApi().dataset_info(repo).sha
        ds = load_dataset(repo, config, split=split, revision=revision, cache_dir='/cache/hf/datasets')
        n = len(ds) if name == 'sib200_twi' else min(limit, len(ds))
        snapshots[name] = [protocol.normalize(name, ds[i], i) for i in range(n)]
        sources[name] = {'repo': repo, 'revision': revision, 'config': config, 'split': split,
                         'available': len(ds), 'selected': n, 'selection': 'first_n_in_source_order',
                         'subjects': dict(Counter(r['subject'] for r in snapshots[name]))}
        print(f'Frozen {name}: {n}/{len(ds)}', flush=True)
    adapters = {variant: {'directory': folder, 'files': {
        name: sha(Path(folder) / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}}
        for variant, folder in ADAPTERS.items()}
    manifest = {'protocol': protocol.VERSION, 'run_id': run_id, 'model': MODEL, 'revision': REVISION,
                'adapters': adapters, 'code': code_hashes(), 'limit': limit, 'sources': sources,
                'inputs_sha256': protocol.digest(snapshots), 'legacy': legacy,
                'scoring': 'conditional_sum_log_probability_of_presented_answer_letters',
                'sib_scoring': 'strict_single_label_generation', 'enable_thinking': False,
                'pubmed_context': 'first_three_contexts_then_first_1500_characters',
                'training': False, 'dataset_overlap_audit': 'not_certified_for_pretraining_or_all_training_data'}
    save(root / 'inputs.json', snapshots)
    save(root / 'manifest.json', manifest)
    save(root / 'SEALED.json', {'manifest_sha256': sha(root / 'manifest.json')})
    for name in manifest['code']:
        dest = root / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    train_vol.commit()
    return {'run_id': run_id, 'root': str(root), 'manifest_sha256': sha(root / 'manifest.json'),
            'counts': {k: len(v) for k, v in snapshots.items()}}


def load_model(variant, manifest):
    import torch
    from transformers import Qwen3_5ForCausalLM, AutoTokenizer
    from peft import PeftModel
    protocol.require(variant in protocol.VARIANTS, 'Unknown variant')
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir='/cache/hf', padding_side='left')
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, revision=REVISION, cache_dir='/cache/hf',
                                              torch_dtype=torch.bfloat16, device_map='auto')
    if variant != 'base':
        spec = manifest['adapters'][variant]
        for name, checksum in spec['files'].items():
            protocol.require(sha(Path(spec['directory']) / name) == checksum, 'Adapter identity changed')
        model = PeftModel.from_pretrained(model, spec['directory'])
    model.eval()
    return model, tok


def prefix(tok, prompt):
    return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def score_mcq(model, tok, row):
    import torch
    prompt_ids, continuations = protocol.answer_tokens(tok, prefix(tok, row['prompt']), row['labels'])
    device = model.device
    scores = {}
    with torch.inference_mode():
        if all(len(ids) == 1 for ids in continuations.values()):
            inputs = torch.tensor([prompt_ids], device=device)
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), logits_to_keep=1).logits[0, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            scores = {label: logprobs[ids[0]].item() for label, ids in continuations.items()}
        else:
            for label, tokens in continuations.items():
                ids = torch.tensor([prompt_ids + tokens[:-1]], device=device)
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=len(tokens)).logits[0].float()
                logprobs = torch.log_softmax(logits, dim=-1)
                scores[label] = sum(logprobs[i, token].item() for i, token in enumerate(tokens))
    return {'prediction': protocol.choose(scores, row['labels']), 'scores': scores,
            'answer_token_ids': continuations, 'prompt_tokens': len(prompt_ids)}


def score_sib(model, tok, row):
    import torch
    from transformers import GenerationConfig
    inputs = tok(prefix(tok, row['prompt']), add_special_tokens=False, return_tensors='pt').to(model.device)
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=64, repetition_penalty=1.0,
                              eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, use_cache=True)
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    text = tok.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return {'prediction': protocol.classify_output(text), 'raw_output': text,
            'legacy_substring_match': row['gold'].lower() in text.lower(),
            'prompt_tokens': inputs['input_ids'].shape[1]}


@app.function(gpu='H100', image=image, volumes=VOLUMES, timeout=14400, retries=0, max_containers=3)
def benchmark(run_id: str, variant: str, expected_sha: str):
    import torch
    protocol.require(variant in protocol.VARIANTS, 'Unknown variant')
    train_vol.reload()
    root = run_path(run_id)
    protocol.require(sha(root / 'manifest.json') == expected_sha, 'Manifest identity changed')
    manifest, inputs = read(root / 'manifest.json'), read(root / 'inputs.json')
    protocol.require(manifest['code'] == code_hashes() and protocol.digest(inputs) == manifest['inputs_sha256'],
                     'Code or frozen input changed')
    result_path = root / f'{variant}.json'
    if result_path.exists() and read(result_path).get('state') == 'completed':
        return read(result_path)
    started = time.time()
    save(result_path, {'variant': variant, 'state': 'loading', 'manifest_sha256': expected_sha})
    train_vol.commit()
    torch.manual_seed(17)
    model, tok = load_model(variant, manifest)
    result = {'protocol': protocol.VERSION, 'variant': variant, 'state': 'running', 'manifest_sha256': expected_sha,
              'model': MODEL, 'revision': REVISION, 'adapter': manifest['adapters'].get(variant),
              'runtime': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'datasets')},
              'tokenizer_sha256': protocol.digest(tok.get_vocab()),
              'chat_template_sha256': protocol.digest(tok.chat_template), 'benchmarks': {}}
    try:
        for name, rows in inputs.items():
            folder = root / 'predictions' / variant / name
            folder.mkdir(parents=True, exist_ok=True)
            evidence = []
            for row in rows:
                path = folder / f'{row["source_index"]:06d}.json'
                if path.exists():
                    item = read(path)
                    protocol.require(item['input_sha256'] == row['input_sha256'] and
                                     item['manifest_sha256'] == expected_sha and item['variant'] == variant,
                                     'Resume evidence differs')
                else:
                    begin = time.time()
                    output = score_sib(model, tok, row) if name == 'sib200_twi' else score_mcq(model, tok, row)
                    item = {**row, **output, 'correct': output['prediction'] == row['gold'],
                            'variant': variant, 'manifest_sha256': expected_sha, 'seconds': time.time() - begin}
                    save(path, item)
                evidence.append(item)
                if len(evidence) % 25 == 0:
                    train_vol.commit()
            result['benchmarks'][name] = protocol.summarize(evidence)
            result['benchmarks'][name]['subjects'] = {subject: protocol.summarize([r for r in evidence if r['subject'] == subject])
                                                    for subject in sorted({r['subject'] for r in evidence})}
            if name == 'sib200_twi':
                result['benchmarks'][name]['legacy_substring_correct'] = sum(r['legacy_substring_match'] for r in evidence)
            save(result_path, result)
            train_vol.commit()
            print(f'{variant} {name}: {result["benchmarks"][name]}', flush=True)
        result['state'] = 'completed'
    except Exception as exc:
        result.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['seconds'] = time.time() - started
        save(result_path, result)
        train_vol.commit()
    return result
