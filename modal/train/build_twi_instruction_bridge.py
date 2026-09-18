"""Transfer a compatible instruction delta onto Twi CPT weights on CPU.

This is task-vector merging, not SFT. No private corpus is uploaded. All tensor
keys, shapes and tokenizer IDs must agree; never merge unrelated architectures.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("ghana-twi-instruction-bridge")
volume = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.7.1", "transformers==5.3.0", "accelerate==1.10.1",
    "peft==0.18.0", "huggingface_hub==1.3.0", "safetensors==0.6.2",
)
SOURCES = {
    "language": ("McGill-NLP/AfriqueQwen3.5-4B-50Langs", "ea443ca5e6674e17c271fb66e54e3282fe78d21a"),
    "instruction": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
}


@app.function(image=image, cpu=4, memory=24576, timeout=2100, max_containers=1,
              retries=0, volumes={"/cache": volume})
def build():
    import hashlib
    import shutil
    import time
    from contextlib import ExitStack
    from datetime import datetime, timezone

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    started = time.monotonic()
    paths = {}
    configs = {}
    vocab = None
    for role, (model, revision) in SOURCES.items():
        paths[role] = Path(snapshot_download(model, revision=revision, cache_dir="/cache/hf",
                          allow_patterns=["*.safetensors", "*.json", "*.jinja", "LICENSE*"]))
        configs[role] = json.loads((paths[role] / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(paths[role], local_files_only=True, trust_remote_code=False)
        if vocab is not None and tokenizer.get_vocab() != vocab:
            raise ValueError("Tokenizer IDs differ; merge refused")
        vocab = tokenizer.get_vocab()
    dimensions = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "vocab_size", "layer_types", "linear_key_head_dim", "linear_value_head_dim")
    for key in dimensions:
        values = [config["text_config"].get(key) for config in configs.values()]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Incompatible architecture: {key}")
    if not all(c.get("tie_word_embeddings") for c in configs.values()):
        raise ValueError("This bounded merge requires tied embeddings")

    report = {"created": datetime.now(timezone.utc).isoformat(), "sources": SOURCES,
              "method": "language + alpha * (instruction - base)", "type": "unvalidated task-vector merge",
              "tokenizer_compatible": True, "outputs": []}
    with ExitStack() as stack:
        tensors = {}
        for role, path in paths.items():
            index = {}
            for file in sorted(path.glob("*.safetensors")):
                handle = stack.enter_context(safe_open(file, framework="pt", device="cpu"))
                for key in handle.keys():
                    if ".language_model." in key:
                        if key in index:
                            raise ValueError("Duplicate tensor key")
                        index[key] = handle
            if not index:
                raise ValueError("No recognized language tensors")
            tensors[role] = index
        keys = sorted(tensors["base"])
        if any(set(index) != set(keys) for index in tensors.values()):
            raise ValueError("Language tensor names differ; merge refused")
        for key in keys:
            shapes = [index[key].get_slice(key).get_shape() for index in tensors.values()]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError(f"Shape mismatch: {key}")

        for alpha in (0.5, 1.0):
            folder = Path(f"/cache/merged/twi-instruction-bridge-a{alpha:.1f}-v1")
            if folder.exists():
                raise FileExistsError("Never overwrite an existing candidate")
            folder.mkdir(parents=True)
            weight_map, hashes, shard, size, total, number = {}, {}, {}, 0, 0, 0

            def flush():
                nonlocal shard, size, number
                if not shard:
                    return
                name = f"model-{number:05d}.safetensors"
                file = folder / name
                save_file(shard, file, metadata={"format": "pt"})
                with file.open("rb") as handle:
                    hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
                weight_map.update({k: name for k in shard})
                number += 1
                shard, size = {}, 0

            for key in keys:
                base = tensors["base"][key].get_tensor(key).float()
                delta = tensors["instruction"][key].get_tensor(key).float().sub_(base)
                del base
                merged = tensors["language"][key].get_tensor(key).float().add_(delta, alpha=alpha)
                del delta
                if not torch.isfinite(merged).all():
                    raise ValueError(f"Non-finite merged weight: {key}")
                value = merged.to(torch.bfloat16).contiguous()
                del merged
                shard[key] = value
                size += value.numel() * value.element_size()
                total += value.numel() * value.element_size()
                if size >= 512 * 1024**2:
                    flush()
            flush()
            (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
            for file in paths["instruction"].iterdir():
                if file.suffix in (".json", ".jinja") and not file.name.startswith("model.safetensors"):
                    shutil.copyfile(file, folder / file.name)
            details = {"alpha": alpha, "path": str(folder), "tensor_count": len(keys), "bytes": total, "sha256": hashes}
            (folder / "merge.json").write_text(json.dumps({**report, **details}, indent=2))
            (folder / "README.md").write_text(
                "# Twi Instruction Bridge, Private Research\n\n"
                f"Unvalidated task-vector merge, alpha={alpha}. Not an SFT checkpoint or clinical model.\n\n"
                "Sources: McGill-NLP/AfriqueQwen3.5-4B-50Langs (CC-BY-4.0), Qwen/Qwen3.5-4B "
                "and Qwen/Qwen3.5-4B-Base (Apache-2.0). Preserve upstream attribution and terms.\n\n"
                "Text-only weights; serve with language_model_only=True. No evaluation or improvement is claimed. "
                "Pinned revisions, merge parameters and integrity hashes are in merge.json. No private training data was used.\n"
            )
            report["outputs"].append(details)
            print(f"Merged alpha={alpha}, tensors={len(keys)}, bytes={total}", flush=True)
            volume.commit()
    report["elapsed_seconds"] = time.monotonic()-started
    return report


@app.local_entrypoint()
def main():
    report = build.remote()
    folder = Path(__file__).resolve().parents[2] / "tmp/general-foundation-comparison"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "bridge-build.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
