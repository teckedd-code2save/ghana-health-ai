#!/usr/bin/env python3
"""Offline, hash-bound PEFT-to-MLX adapter conversion; no cloud inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REVISION = "693f12e444a30971ca53889128d7bade9491ecb3"
BASE_SHA = "6c03034e336dc916d46b353493bc30b3244f83b80d487be83d6ad0999af15af6"
VARIANTS = {
    "r7": ("understanding-adaptation-r7/understanding_20261118T010701Z/responses-adapter",
           "f6e6b484ca404146016eefabd170738e66791880775238e435e31577300e57a7",
           "d155906d68654532d884d3ce65971248249df7e4a7e1514b8635bc40f07ed5c2"),
    "r6": ("understanding-adaptation-r6/understanding_20261113T152209Z/responses-adapter",
           "6e56d8c7c04f456f94579d4bd82064521c97d0b9e23dbbef1e9783280568861c",
           "577b80656dbcff84886c4e402022d9507abbef34b0e0013bceab029ccf4c9403"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def map_key(key):
    match = re.fullmatch(r"base_model\.model\.(model\.layers\.(\d+)\.(?:linear_attn|self_attn|mlp)\.\w+)\.lora_([AB])\.weight", key)
    require(match is not None and int(match[2]) < 32, "Unexpected PEFT tensor: " + key)
    return "language_model." + match[1] + ".lora_" + match[3].lower()


def validate_config(config):
    require(config["base_model_name_or_path"] == MODEL and config["revision"] == REVISION, "Wrong base pin")
    require(config["peft_type"] == "LORA" and config["r"] == 16 and config["lora_alpha"] == 32, "Unexpected adapter scale")
    for key in ("use_dora", "use_rslora", "fan_in_fan_out", "rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias"):
        require(not config.get(key), "Unsupported PEFT behavior: " + key)
    require(config["bias"] == "none", "Unexpected trained bias")


def convert_adapters(export_root, bundle, out):
    import mlx.core as mx
    import numpy as np
    from safetensors.numpy import load_file
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    from mlx_lm.tuner.utils import load_adapters
    from mlx.utils import tree_flatten
    from check_afrique_training_contract import ASSET_HASHES

    require(not out.exists(), "Use a new output directory")
    for name, checksum in ASSET_HASHES.items():
        require(sha(bundle / "tokenizer" / name) == checksum, "Changed frozen asset: " + name)
    config = json.loads((bundle / "tokenizer/config.json").read_text())
    out.mkdir(parents=True, mode=0o700)
    receipt = {"base": MODEL, "revision": REVISION, "base_sha256": BASE_SHA,
               "runtime": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "numpy", "transformers")},
               "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
               "variants": {}, "cloud_compute_started": False,
               "claims": "Adapter mapping and numerical checks only; not full-model parity or a quality evaluation"}
    for variant, (relative, weight_sha, config_sha) in VARIANTS.items():
        source = export_root / relative
        require(sha(source / "adapter_model.safetensors") == weight_sha, "Source adapter changed")
        require(sha(source / "adapter_config.json") == config_sha, "Source adapter config changed")
        peft = json.loads((source / "adapter_config.json").read_text())
        validate_config(peft)
        original = load_file(source / "adapter_model.safetensors")
        require(len(original) == 496 and sum(v.size for v in original.values()) == 43278336, "Incomplete adapter")
        require(all(np.isfinite(v).all() for v in original.values()), "Nonfinite adapter")
        converted = {map_key(k): mx.array(v.T.copy()) for k, v in original.items()}
        require(len(converted) == len(original), "Colliding tensor map")
        keys = sorted({".".join(k.split(".")[4:-1]) for k in converted})
        destination = out / variant
        destination.mkdir()
        save(destination / "adapter_config.json", {"fine_tune_type": "lora", "num_layers": 32,
             "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.0, "keys": keys}})
        mx.save_safetensors(str(destination / "adapters.safetensors"), converted)

        # Shapes are inspected lazily: no 9B base weights are allocated or evaluated.
        shape_model = Model(ModelArgs.from_dict(config))
        base_shapes = {k: v.shape for k, v in tree_flatten(shape_model.parameters())}
        for key, a in original.items():
            if key.endswith(".lora_A.weight"):
                b = original[key.replace(".lora_A.weight", ".lora_B.weight")]
                base_key = map_key(key).removesuffix(".lora_a") + ".weight"
                require(a.shape[0] == b.shape[1] == 16 and
                        base_shapes.get(base_key) == (b.shape[0], a.shape[1]),
                        "Adapter dimensions differ from base projection: " + key)
        load_adapters(shape_model, destination)
        params = dict(tree_flatten(shape_model.parameters()))
        adapter_names = {k for k in params if k.endswith((".lora_a", ".lora_b"))}
        require(adapter_names == set(converted), "Missing or unexpected target modules")
        for key, value in converted.items():
            require(params[key].shape == value.shape, "Adapter shape differs: " + key)
        del shape_model, params
        max_error = 0.0
        for key, a in original.items():
            if not key.endswith(".lora_A.weight"):
                continue
            b = original[key.replace(".lora_A.weight", ".lora_B.weight")]
            x = np.sin(np.arange(2 * a.shape[1], dtype=np.float32).reshape(2, -1)) / 10
            reference = ((x @ a.T) @ b.T) * 2.0
            observed = np.array((mx.array(x) @ converted[map_key(key)]) @
                                converted[map_key(key.replace(".lora_A.weight", ".lora_B.weight"))] * 2.0)
            require(np.allclose(observed, reference, atol=1e-4, rtol=2e-4), "LoRA delta parity failed: " + key)
            max_error = max(max_error, float(np.max(np.abs(observed - reference))))
        receipt["variants"][variant] = {"source_weights_sha256": weight_sha,
            "source_config_sha256": config_sha, "converted_weights_sha256": sha(destination / "adapters.safetensors"),
            "converted_config_sha256": sha(destination / "adapter_config.json"),
            "tensor_count": len(converted), "modules_checked": len(converted) // 2,
            "parameter_count": 43278336, "scale": 2.0, "max_delta_absolute_error": max_error,
            "all_target_shapes_verified": True}
        del original, converted
        mx.clear_cache()
        print(json.dumps({"adapter_verified": variant, **receipt["variants"][variant]}), flush=True)
    save(out / "manifest.json", receipt)
    return receipt


def convert_base(source, out):
    from mlx_lm.convert import convert
    import mlx.core as mx
    from check_afrique_training_contract import ASSET_HASHES
    for name, checksum in ASSET_HASHES.items():
        require(sha(source / name) == checksum, "Base metadata differs: " + name)
    require(sha(source / "model.safetensors") == BASE_SHA, "Base file hash differs or download incomplete")
    require(not out.exists(), "Use a new quantized-base directory")
    mx.set_memory_limit(8 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    convert(str(source), str(out), quantize=True, q_bits=4, q_group_size=64, q_mode="affine")
    save(out / "source-identity.json", {"model": MODEL, "revision": REVISION, "source_sha256": BASE_SHA,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        "cloud_compute_started": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    adapters = commands.add_parser("adapters")
    adapters.add_argument("--export-root", type=Path, required=True)
    adapters.add_argument("--bundle", type=Path, required=True)
    adapters.add_argument("--out", type=Path, required=True)
    base = commands.add_parser("base")
    base.add_argument("--source", type=Path, required=True)
    base.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "adapters":
        convert_adapters(args.export_root, args.bundle, args.out)
    else:
        convert_base(args.source, args.out)


if __name__ == "__main__":
    main()
