"""
Twi understand SFT — research path.

Data sources (priority):
  1. --use-local-silver → data/understanding-corpus/silver-medical-plus-language-v1
  2. --dataset HF chat JSONL (messages column)
  3. --use-ghananlp-parallel → Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT
     converted to Twi-first health-style chat turns

  modal run modal/train/train_understand.py --use-local-silver --smoke
  modal run --detach modal/train/train_understand.py \\
    --use-local-silver --max-steps 500 \\
    --push-repo teckedd/gha-understand-twi-medical-plus-language-v3

Language policy: this model is for semantic recovery, not direct medical advice.
"""

from __future__ import annotations

import os
import json
import random
from typing import Any, Optional

import modal

app = modal.App("ghana-health-understand-train")
vol = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_LOCAL_SILVER_DIR = os.path.join(
    _REPO_ROOT,
    "data",
    "understanding-corpus",
    "silver-medical-plus-language-v1",
)
_REMOTE_SILVER_DIR = "/root/gha_understanding_silver_medical_plus_language_v1"
_OUTPUT_SUFFIX = "medical_plus_language_v1"
_DEFAULT_PUSH_REPO = "teckedd/gha-understand-twi-medical-plus-language-v3"
_VALID_HF_DATASETS = [
    "ghananlpcommunity/ghana-health-symptoms:cc-by-nc-4.0",
    "google/WaxalNLP:language coverage rows",
]
_SOURCE_NOTES = [
    "Ghana Health Symptoms rows are the primary medical semantic-recovery source.",
    "WAXAL rows are included only as language-coverage silver rows after filtering.",
    "GhanaNLP speech-text rows are included as local language-coverage silver rows; "
    "they are described in the card body, not HF front-matter, until the source id "
    "and license are audited.",
    "Product-failure seed rows cover observed app failures such as eye pain, child fever, "
    "hospital navigation, malaria follow-ups, and Twi commerce purchase/search requests.",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "peft==0.13.2",
        "trl==0.12.1",
        "bitsandbytes==0.44.1",
        "huggingface_hub==0.26.2",
    )
    .add_local_file(
        local_path=os.path.join(_TRAIN_DIR, "model_card.py"),
        remote_path="/root/gha_train/model_card.py",
    )
    .add_local_dir(
        local_path=_LOCAL_SILVER_DIR,
        remote_path=_REMOTE_SILVER_DIR,
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


SYSTEM_TWI = (
    "Wo yɛ Ghana Health AI. Ka Twi. English only if user prefers English. "
    "Nnyɛ oduruyɛfoɔ. No bare CHW/ANC."
)
SYSTEM_EN = (
    "You are Ghana Health AI. User prefers English. "
    "Not a doctor. Expand acronyms. Short spoken answers."
)


def _parallel_to_messages(row: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """Map GhanaNLP parallel row → chat messages. Prefer Twi as assistant language."""
    # Common column names across Ghana-NLP parallel releases
    tw = None
    en = None
    for k, v in row.items():
        kl = k.lower()
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if kl in ("twi", "tw", "aka", "asante", "akuapem", "target", "translation") or "twi" in kl:
            tw = s
        if kl in ("english", "en", "source", "eng") or kl.startswith("en"):
            en = s
    # fallback: first two string columns
    if not tw or not en:
        strs = [str(v).strip() for v in row.values() if isinstance(v, str) and str(v).strip()]
        if len(strs) >= 2:
            en, tw = strs[0], strs[1]

    if not tw:
        return None

    # ~85% Twi path: user may speak EN or Twi, assistant answers Twi
    prefer_en = rng.random() < 0.15 and bool(en)
    if prefer_en:
        user = en or tw
        assistant = en
        system = SYSTEM_EN
        lang = "en"
    else:
        # User utters Twi (or English question); assistant Twi
        if en and rng.random() < 0.35:
            user = en  # code-mix / EN user still gets Twi answer
        else:
            user = tw
        assistant = tw
        system = SYSTEM_TWI
        lang = "tw"

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "language": lang,
        "source": "ghananlp_parallel",
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    volumes={"/data": vol},
    secrets=SECRETS,
)
def train(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    dataset_name: str = "",
    use_local_silver: bool = True,
    use_ghananlp_parallel: bool = False,
    max_steps: int = 500,
    learning_rate: float = 2e-4,
    push_repo: Optional[str] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/data/hf"
    out_dir = f"/data/sft/{base_model.replace('/', '_')}_{_OUTPUT_SUFFIX}"
    os.makedirs(out_dir, exist_ok=True)

    if smoke:
        max_steps = min(max_steps, 15)

    from datasets import Dataset, load_dataset

    rows: list[dict[str, Any]] = []
    source = dataset_name or "none"
    datasets_used: list[str] = []

    if use_local_silver:
        local_train = os.path.join(_REMOTE_SILVER_DIR, "train.jsonl")
        if os.path.exists(local_train):
            with open(local_train, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "messages" in row and row["messages"]:
                        rows.append({"messages": row["messages"]})
            source = "local:understanding-corpus/silver-medical-plus-language-v1"
            datasets_used.extend(_VALID_HF_DATASETS)
        else:
            print(f"[understand-train] local silver dataset missing: {local_train}")

    if use_ghananlp_parallel or (not dataset_name and not rows):
        # Research default: Ghana-NLP Twi↔EN parallel
        pname = "Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT"
        try:
            raw = load_dataset(pname, token=token, cache_dir=cache)
            split = "train" if "train" in raw else list(raw.keys())[0]
            rng = random.Random(42)
            for row in raw[split]:
                m = _parallel_to_messages(dict(row), rng)
                if m:
                    rows.append(m)
            source = pname
            datasets_used.append(pname)
        except Exception as exc:  # noqa: BLE001
            if not dataset_name:
                return {
                    "status": "error",
                    "message": f"Failed to load GhanaNLP parallel: {exc}",
                    "hint": "Pass --dataset with messages JSONL or fix HF token",
                }

    if dataset_name:
        ds = load_dataset(dataset_name, token=token, cache_dir=cache)
        split = "train" if "train" in ds else list(ds.keys())[0]
        for row in ds[split]:
            if "messages" in row:
                rows.append({"messages": row["messages"]})
            else:
                rng = random.Random(hash(str(row)) % 10_000)
                m = _parallel_to_messages(dict(row), rng)
                if m:
                    rows.append(m)
        source = f"{source}+{dataset_name}" if source != "none" else dataset_name
        datasets_used.append(dataset_name)

    if len(rows) < 8:
        return {
            "status": "awaiting_data",
            "message": "Need chat messages or GhanaNLP parallel rows",
            "n": len(rows),
            "research": "docs/research-stack.md",
        }

    if smoke:
        rows = rows[:24]

    train_ds = Dataset.from_list(rows)

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Public models often fail with a restricted/expired HF token on Xet CDN (403).
    # Prefer anonymous download for open bases; only pass token when needed.
    def _load_kwargs():
        kw: dict[str, Any] = {"cache_dir": cache}
        if token:
            kw["token"] = token
        return kw

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, **_load_kwargs())
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache, token=None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
            **_load_kwargs(),
        )
    except Exception as first_exc:  # noqa: BLE001
        print(f"[understand-train] load with token failed: {first_exc}; retry anonymous")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=cache,
            torch_dtype="auto",
            device_map="auto",
            token=None,
        )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = SFTConfig(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        logging_steps=5,
        save_steps=max(50, max_steps // 5),
        bf16=True,
        push_to_hub=bool(push_repo) and not smoke,
        hub_model_id=push_repo,
        hub_token=token,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    train_output = trainer.train()
    train_metrics = dict(getattr(train_output, "metrics", {}) or {})
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    vol.commit()

    hub_status = None
    if push_repo and not smoke and token:
        try:
            import sys

            sys.path.insert(0, "/root/gha_train")
            from model_card import write_and_push_model_card  # type: ignore

            write_and_push_model_card(
                push_repo,
                task="text-generation",
                language=["tw", "ak", "en"],
                base_model=base_model,
                datasets=datasets_used or [source],
                metrics={
                    "n_train": len(train_ds),
                    "max_steps": max_steps,
                    **(
                        {"train_loss": float(train_metrics["train_loss"])}
                        if "train_loss" in train_metrics
                        else {}
                    ),
                },
                summary=(
                    "Twi/Akan semantic-recovery LoRA for Ghana Health AI. "
                    "Trained on large medical plus language-coverage silver corpus for research evaluation."
                ),
                extra_markdown=(
                    "## Dataset status\n\n"
                    "This is a research checkpoint trained from machine annotations. "
                    "Rows are not human-gold labels. The main medical source is CC-BY-NC-4.0, "
                    "so use is non-commercial research unless separate permission is obtained.\n\n"
                    "## Source notes\n\n"
                    + "\n".join(f"- {note}" for note in _SOURCE_NOTES)
                    + "\n"
                ),
                license_id="cc-by-nc-4.0" if any("cc-by-nc" in d for d in datasets_used) else "apache-2.0",
                tags=["lora", "sft", "twi", "ghana-nlp", "semantic-recovery", "silver-corpus"],
                pipeline_tag="text-generation",
                token=token,
            )
            hub_status = f"pushed:{push_repo}+card"
        except Exception as exc:  # noqa: BLE001
            hub_status = f"push_failed:{exc}"

    return {
        "status": "ok",
        "source": source,
        "n_train": len(train_ds),
        "base_model": base_model,
        "output_dir": out_dir,
        "push_repo": push_repo,
        "hub": hub_status,
        "train_metrics": train_metrics,
        "research": "docs/research-stack.md",
    }


@app.function(
    image=image,
    timeout=30 * 60,
    volumes={"/data": vol},
    secrets=SECRETS,
)
def push_saved(
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    push_repo: str = _DEFAULT_PUSH_REPO,
    train_loss: Optional[float] = None,
) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return {"status": "error", "message": "Missing HF_TOKEN secret"}

    out_dir = f"/data/sft/{base_model.replace('/', '_')}_{_OUTPUT_SUFFIX}"
    if not os.path.isdir(out_dir):
        parent = os.path.dirname(out_dir)
        available = os.listdir(parent) if os.path.isdir(parent) else []
        return {
            "status": "missing_model",
            "output_dir": out_dir,
            "available": available,
        }

    from huggingface_hub import HfApi
    import sys

    sys.path.insert(0, "/root/gha_train")
    from model_card import write_and_push_model_card  # type: ignore

    api = HfApi(token=token)
    api.create_repo(repo_id=push_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=out_dir,
        repo_id=push_repo,
        repo_type="model",
        token=token,
        commit_message="model: upload medical plus language adapter",
    )
    write_and_push_model_card(
        push_repo,
        task="text-generation",
        language=["tw", "ak", "en"],
        base_model=base_model,
        datasets=_VALID_HF_DATASETS,
        metrics={
            "n_train": 6329,
            "max_steps": 650,
            **({"train_loss": train_loss} if train_loss is not None else {}),
        },
        summary=(
            "Twi/Akan semantic-recovery LoRA for Ghana Health AI. "
            "Trained on a cleaned medical plus language-coverage silver corpus for research evaluation."
        ),
        extra_markdown=(
            "## Dataset status\n\n"
            "This is a research checkpoint trained from machine annotations. "
            "Rows are not human-gold labels. The main medical source is CC-BY-NC-4.0, "
            "so use is non-commercial research unless separate permission is obtained.\n\n"
            "## Source notes\n\n"
            + "\n".join(f"- {note}" for note in _SOURCE_NOTES)
            + "\n\n"
            "## Training run\n\n"
            "- Train rows: `6329`\n"
            "- Steps: `650`\n"
            "- Corpus: `data/understanding-corpus/silver-medical-plus-language-v1`\n"
            + (f"- Final train loss: `{train_loss:.4f}`\n" if train_loss is not None else "")
        ),
        license_id="cc-by-nc-4.0",
        tags=["lora", "sft", "twi", "ghana-nlp", "semantic-recovery", "silver-corpus"],
        pipeline_tag="text-generation",
        token=token,
    )
    return {
        "status": "pushed",
        "repo": push_repo,
        "url": f"https://huggingface.co/{push_repo}",
        "output_dir": out_dir,
    }


@app.local_entrypoint()
def main(
    dataset: str = "",
    use_local_silver: bool = True,
    use_ghananlp_parallel: bool = False,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    smoke: bool = False,
    push_repo: str = "",
    max_steps: int = 500,
    push_only: bool = False,
    train_loss: Optional[float] = None,
):
    if push_only:
        print(
            push_saved.remote(
                base_model=base_model,
                push_repo=push_repo or _DEFAULT_PUSH_REPO,
                train_loss=train_loss,
            )
        )
        return

    print(
        train.remote(
            dataset_name=dataset,
            use_local_silver=use_local_silver,
            use_ghananlp_parallel=use_ghananlp_parallel,
            base_model=base_model,
            smoke=smoke,
            push_repo=push_repo or None,
            max_steps=max_steps,
        )
    )
