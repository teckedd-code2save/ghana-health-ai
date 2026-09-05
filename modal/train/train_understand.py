"""
Twi understand SFT — research path.

Data sources (priority):
  1. --use-local-silver → data/understanding-corpus/silver-medical-paired-v2
  2. --dataset HF chat JSONL (messages column)
  3. --use-ghananlp-parallel → Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT
     converted to Twi-first health-style chat turns

  modal run modal/train/train_understand.py --use-local-silver --smoke
  modal run --detach modal/train/train_understand.py \\
    --use-local-silver --max-steps 500 \\
    --push-repo teckedd/gha-understand-twi-medical-v4

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
    "silver-medical-paired-v2",
)
_REMOTE_SILVER_DIR = "/root/gha_understanding_silver_medical_paired_v2"
_OUTPUT_SUFFIX = "medical_paired_v2"
_DEFAULT_PUSH_REPO = "teckedd/gha-understand-twi-medical-v4"
_VALID_HF_DATASETS = [
    "ghananlpcommunity/ghana-health-symptoms",
]
_SOURCE_NOTES = [
    "Exactly 7,000 Ghana Health Symptoms rows are used: 5,659 train, 669 dev, and 672 test.",
    "The Twi and concise English meanings are source-paired silver labels, not human-gold annotations.",
    "WAXAL, GhanaNLP speech rows, synthetic prompt seeds, and product-failure fixtures are not mixed into this release.",
    "The source is CC-BY-NC-4.0, so this checkpoint is non-commercial research only.",
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
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    dataset_name: str = "",
    use_local_silver: bool = True,
    use_ghananlp_parallel: bool = False,
    max_steps: int = 500,
    learning_rate: float = 1e-4,
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
            source = "local:understanding-corpus/silver-medical-paired-v2"
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

    eval_rows: list[dict[str, Any]] = []
    if use_local_silver:
        local_dev = os.path.join(_REMOTE_SILVER_DIR, "dev.jsonl")
        if os.path.exists(local_dev):
            with open(local_dev, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "messages" in row and row["messages"]:
                        eval_rows.append({"messages": row["messages"]})

    if smoke:
        rows = rows[:24]
        eval_rows = eval_rows[:8]

    train_ds = Dataset.from_list(rows)
    eval_ds = Dataset.from_list(eval_rows) if eval_rows else None

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
        save_steps=max(25, max_steps // 5),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=max(25, max_steps // 5) if eval_ds is not None else None,
        bf16=True,
        push_to_hub=bool(push_repo) and not smoke,
        hub_model_id=push_repo,
        hub_token=token,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    train_output = trainer.train()
    train_metrics = dict(getattr(train_output, "metrics", {}) or {})
    if eval_ds is not None:
        train_metrics.update(trainer.evaluate())
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
                    "n_eval": len(eval_ds) if eval_ds is not None else 0,
                    "max_steps": max_steps,
                    **(
                        {"train_loss": float(train_metrics["train_loss"])}
                        if "train_loss" in train_metrics
                        else {}
                    ),
                    **(
                        {"eval_loss": float(train_metrics["eval_loss"])}
                        if "eval_loss" in train_metrics
                        else {}
                    ),
                },
                summary=(
                    "Twi/Akan medical semantic-recovery LoRA for Ghana Health AI. "
                    "Trained on a 7,000-row paired medical silver corpus for research evaluation."
                ),
                extra_markdown=(
                    "## Dataset status\n\n"
                    "This is a research checkpoint trained from source-paired silver labels. "
                    "Rows are not human-gold annotations. The medical source is CC-BY-NC-4.0, "
                    "so use is non-commercial research unless separate permission is obtained.\n\n"
                    "## Source notes\n\n"
                    + "\n".join(f"- {note}" for note in _SOURCE_NOTES)
                    + "\n"
                ),
                license_id="cc-by-nc-4.0"
                if "ghananlpcommunity/ghana-health-symptoms" in datasets_used
                else "apache-2.0",
                tags=["lora", "sft", "twi", "ghana-nlp", "semantic-recovery", "silver-corpus"],
                pipeline_tag="text-generation",
                intended_use=[
                    "Recover structured Twi/Akan medical meaning for research evaluation.",
                    "Produce English meaning, intent, body-system entities, and ambiguity fields for a downstream response model.",
                ],
                out_of_scope=[
                    "Direct medical advice or patient-facing response generation.",
                    "Clinical diagnosis, triage, or autonomous medical decisions.",
                    "Commercial use of this checkpoint without separate permission for the CC-BY-NC-4.0 source data.",
                ],
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
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    push_repo: str = _DEFAULT_PUSH_REPO,
    train_loss: Optional[float] = None,
    eval_loss: Optional[float] = None,
    eval_passed: Optional[int] = None,
    eval_total: Optional[int] = None,
    eval_failed_cases: str = "",
    eval_health_passed: Optional[int] = None,
    eval_health_total: Optional[int] = None,
    eval_commerce_passed: Optional[int] = None,
    eval_commerce_total: Optional[int] = None,
    test_total: Optional[int] = None,
    test_parse_passed: Optional[int] = None,
    test_intent_passed: Optional[int] = None,
    test_body_system_passed: Optional[int] = None,
    test_strict_passed: Optional[int] = None,
    test_mean_english_f1: Optional[float] = None,
    baseline_eval_passed: Optional[int] = None,
    baseline_eval_total: Optional[int] = None,
    baseline_test_total: Optional[int] = None,
    baseline_test_parse_passed: Optional[int] = None,
    baseline_test_intent_passed: Optional[int] = None,
    baseline_test_body_system_passed: Optional[int] = None,
    baseline_test_strict_passed: Optional[int] = None,
    baseline_test_mean_english_f1: Optional[float] = None,
    train_rows: int = 5659,
    dev_rows: int = 669,
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
        commit_message="model: upload paired medical understanding adapter",
        ignore_patterns=["checkpoint-*"],
        delete_patterns=["checkpoint-*/*"],
    )
    eval_metrics: dict[str, Any] = {}
    eval_markdown = ""
    if eval_passed is not None and eval_total:
        eval_rate = float(eval_passed) / float(eval_total)
        failed_case_ids = [case.strip() for case in eval_failed_cases.split(",") if case.strip()]
        eval_metrics = {
            "product_fixture_pass_rate": eval_rate,
            "product_fixture_passed": eval_passed,
            "product_fixture_total": eval_total,
            **(
                {"health_fixture_pass_rate": float(eval_health_passed) / float(eval_health_total)}
                if eval_health_passed is not None and eval_health_total
                else {}
            ),
            **(
                {"commerce_fixture_pass_rate": float(eval_commerce_passed) / float(eval_commerce_total)}
                if eval_commerce_passed is not None and eval_commerce_total
                else {}
            ),
            **(
                {"base_product_fixture_pass_rate": float(baseline_eval_passed or 0) / float(baseline_eval_total)}
                if baseline_eval_total
                else {}
            ),
        }
        eval_payload = {
            "status": "rejected_for_product" if eval_passed < eval_total else "passed",
            "promotion_decision": "do_not_promote" if eval_passed < eval_total else "eligible_for_review",
            "passed": eval_passed,
            "total": eval_total,
            "pass_rate": eval_rate,
            "eval_set": "project product fixtures",
            "note": (
                "Small product-critical fixture evaluation. Failed cases should be treated "
                "as promotion blockers for any product routing, including an opt-in research route."
            ),
            "failed_case_ids": failed_case_ids,
            "by_focus": {
                **(
                    {"health": {"passed": eval_health_passed, "total": eval_health_total}}
                    if eval_health_passed is not None and eval_health_total
                    else {}
                ),
                **(
                    {"commerce": {"passed": eval_commerce_passed, "total": eval_commerce_total}}
                    if eval_commerce_passed is not None and eval_commerce_total
                    else {}
                ),
            },
            **(
                {
                    "base_model_comparison": {
                        "passed": baseline_eval_passed,
                        "total": baseline_eval_total,
                        "pass_rate": float(baseline_eval_passed or 0) / float(baseline_eval_total),
                    }
                }
                if baseline_eval_total
                else {}
            ),
        }
        api.upload_file(
            path_or_fileobj=json.dumps(eval_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            path_in_repo="eval/product-fixtures.v0.json",
            repo_id=push_repo,
            repo_type="model",
            token=token,
            commit_message="eval: add product fixture results",
        )
        eval_markdown = (
            "\n## Product fixture evaluation\n\n"
            f"- Status: `{'rejected_for_product' if eval_passed < eval_total else 'passed'}`\n"
            f"- Passed: `{eval_passed}/{eval_total}`\n"
            f"- Pass rate: `{eval_rate:.2%}`\n"
            + (
                f"- Health fixtures: `{eval_health_passed}/{eval_health_total}`\n"
                if eval_health_passed is not None and eval_health_total
                else ""
            )
            + (
                f"- Commerce fixtures: `{eval_commerce_passed}/{eval_commerce_total}`\n"
                if eval_commerce_passed is not None and eval_commerce_total
                else ""
            )
            + "- Artifact: `eval/product-fixtures.v0.json`\n"
            + (
                f"- Unadapted base comparison: `{baseline_eval_passed or 0}/{baseline_eval_total}`\n"
                if baseline_eval_total
                else ""
            )
            + "- Promotion decision: **do not route product traffic to this checkpoint.**\n"
        )
    test_metrics: dict[str, Any] = {}
    test_markdown = ""
    if test_total:
        def _rate(value: Optional[int]) -> float:
            return float(value or 0) / float(test_total)

        test_metrics = {
            "test_rows": test_total,
            "test_parse_rate": _rate(test_parse_passed),
            "test_intent_accuracy": _rate(test_intent_passed),
            "test_body_system_accuracy": _rate(test_body_system_passed),
            "test_strict_pass_rate": _rate(test_strict_passed),
            **(
                {"test_mean_natural_english_token_f1": test_mean_english_f1}
                if test_mean_english_f1 is not None
                else {}
            ),
        }
        test_payload = {
            "status": "evaluation_only",
            "promotion_decision": "do_not_promote",
            "test_set": "silver-medical-paired-v2/test.jsonl",
            "total": test_total,
            "parse_passed": test_parse_passed,
            "intent_passed": test_intent_passed,
            "body_system_passed": test_body_system_passed,
            "strict_passed": test_strict_passed,
            "mean_natural_english_token_f1": test_mean_english_f1,
            "strict_definition": (
                "parseable JSON, exact intent, exact body system, and natural-English token F1 >= 0.5"
            ),
            "intent_metric_limitation": (
                "Every row has the same health_symptom_report intent, so exact intent accuracy "
                "does not demonstrate intent generalization."
            ),
            **(
                {
                    "base_model_comparison": {
                        "total": baseline_test_total,
                        "parse_passed": baseline_test_parse_passed,
                        "intent_passed": baseline_test_intent_passed,
                        "body_system_passed": baseline_test_body_system_passed,
                        "strict_passed": baseline_test_strict_passed,
                        "mean_natural_english_token_f1": baseline_test_mean_english_f1,
                    }
                }
                if baseline_test_total
                else {}
            ),
        }
        api.upload_file(
            path_or_fileobj=json.dumps(test_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            path_in_repo="eval/held-out-test.v0.json",
            repo_id=push_repo,
            repo_type="model",
            token=token,
            commit_message="eval: add held-out corpus test results",
        )
        test_markdown = (
            "\n## Held-out corpus evaluation\n\n"
            f"- Rows: `{test_total}`\n"
            f"- Parseable JSON: `{test_parse_passed or 0}/{test_total}`\n"
            f"- Exact intent: `{test_intent_passed or 0}/{test_total}`\n"
            "  This is not evidence of intent generalization because all test rows share one intent.\n"
            f"- Exact body system: `{test_body_system_passed or 0}/{test_total}`\n"
            f"- Strict semantic pass: `{test_strict_passed or 0}/{test_total}`\n"
            + (
                f"- Mean natural-English token F1: `{test_mean_english_f1:.4f}`\n"
                if test_mean_english_f1 is not None
                else ""
            )
            + "- Artifact: `eval/held-out-test.v0.json`\n"
            + (
                "- Unadapted base comparison: "
                f"parse `{baseline_test_parse_passed or 0}/{baseline_test_total}`, "
                f"strict `{baseline_test_strict_passed or 0}/{baseline_test_total}`, "
                f"mean English F1 `{baseline_test_mean_english_f1 or 0:.4f}`\n"
                if baseline_test_total
                else ""
            )
            + "- Promotion decision: **do not promote; semantic recovery is not reliable.**\n"
        )
        if baseline_test_total:
            test_metrics.update(
                {
                    "base_test_parse_rate": float(baseline_test_parse_passed or 0)
                    / float(baseline_test_total),
                    "base_test_strict_pass_rate": float(baseline_test_strict_passed or 0)
                    / float(baseline_test_total),
                    **(
                        {"base_test_mean_natural_english_token_f1": baseline_test_mean_english_f1}
                        if baseline_test_mean_english_f1 is not None
                        else {}
                    ),
                }
            )
    write_and_push_model_card(
        push_repo,
        task="text-generation",
        language=["tw", "ak", "en"],
        base_model=base_model,
        datasets=_VALID_HF_DATASETS,
        metrics={
            "n_train": train_rows,
            "n_eval": dev_rows,
            **({"train_loss": train_loss} if train_loss is not None else {}),
            **({"eval_loss": eval_loss} if eval_loss is not None else {}),
            **eval_metrics,
            **test_metrics,
        },
        summary=(
            "Twi/Akan medical semantic-recovery LoRA for Ghana Health AI. "
            "Trained on a 7,000-row paired medical silver corpus for research evaluation."
        ),
        extra_markdown=(
            "## Dataset status\n\n"
            "This is a research checkpoint trained from source-paired silver labels. "
            "Rows are not human-gold annotations. The medical source is CC-BY-NC-4.0, "
            "so use is non-commercial research unless separate permission is obtained.\n\n"
            "## Source notes\n\n"
            + "\n".join(f"- {note}" for note in _SOURCE_NOTES)
            + "\n\n"
            "## Training run\n\n"
            f"- Train rows: `{train_rows}`\n"
            f"- Development rows: `{dev_rows}`\n"
            "- Corpus: `data/understanding-corpus/silver-medical-paired-v2`\n"
            + (f"- Final train loss: `{train_loss:.4f}`\n" if train_loss is not None else "")
            + (f"- Development loss: `{eval_loss:.4f}`\n" if eval_loss is not None else "")
            + eval_markdown
            + test_markdown
            + "\n## Research decision\n\n"
            "This checkpoint is rejected for product use. It learned the JSON schema and the "
            "single medical intent, but it did not learn reliable Twi-to-English semantics and "
            "failed all commerce fixtures. Do not use it to generate or ground patient-facing "
            "responses. The frozen base comparison confirms a measurable medical adaptation, "
            "but not enough semantic reliability for product use. The next experiment requires "
            "a mixed-domain, semantically varied corpus, not additional epochs on this corpus.\n"
        ),
        license_id="cc-by-nc-4.0",
        tags=["lora", "sft", "twi", "ghana-nlp", "semantic-recovery", "silver-corpus"],
        pipeline_tag="text-generation",
        intended_use=[
            "Recover structured Twi/Akan medical meaning for research evaluation.",
            "Produce English meaning, intent, body-system entities, and ambiguity fields for a downstream response model.",
        ],
        out_of_scope=[
            "Direct medical advice or patient-facing response generation.",
            "Clinical diagnosis, triage, or autonomous medical decisions.",
            "Commercial use of this checkpoint without separate permission for the CC-BY-NC-4.0 source data.",
        ],
        token=token,
    )
    return {
        "status": "pushed",
        "repo": push_repo,
        "url": f"https://huggingface.co/{push_repo}",
        "output_dir": out_dir,
        "eval": {**eval_metrics, **test_metrics},
    }


@app.local_entrypoint()
def main(
    dataset: str = "",
    use_local_silver: bool = True,
    use_ghananlp_parallel: bool = False,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    smoke: bool = False,
    push_repo: str = "",
    max_steps: int = 500,
    push_only: bool = False,
    train_loss: Optional[float] = None,
    eval_loss: Optional[float] = None,
    eval_passed: Optional[int] = None,
    eval_total: Optional[int] = None,
    eval_failed_cases: str = "",
    eval_health_passed: Optional[int] = None,
    eval_health_total: Optional[int] = None,
    eval_commerce_passed: Optional[int] = None,
    eval_commerce_total: Optional[int] = None,
    test_total: Optional[int] = None,
    test_parse_passed: Optional[int] = None,
    test_intent_passed: Optional[int] = None,
    test_body_system_passed: Optional[int] = None,
    test_strict_passed: Optional[int] = None,
    test_mean_english_f1: Optional[float] = None,
    baseline_eval_passed: Optional[int] = None,
    baseline_eval_total: Optional[int] = None,
    baseline_test_total: Optional[int] = None,
    baseline_test_parse_passed: Optional[int] = None,
    baseline_test_intent_passed: Optional[int] = None,
    baseline_test_body_system_passed: Optional[int] = None,
    baseline_test_strict_passed: Optional[int] = None,
    baseline_test_mean_english_f1: Optional[float] = None,
    train_rows: int = 5659,
    dev_rows: int = 669,
):
    if push_only:
        print(
            push_saved.remote(
                base_model=base_model,
                push_repo=push_repo or _DEFAULT_PUSH_REPO,
                train_loss=train_loss,
                eval_loss=eval_loss,
                eval_passed=eval_passed,
                eval_total=eval_total,
                eval_failed_cases=eval_failed_cases,
                eval_health_passed=eval_health_passed,
                eval_health_total=eval_health_total,
                eval_commerce_passed=eval_commerce_passed,
                eval_commerce_total=eval_commerce_total,
                test_total=test_total,
                test_parse_passed=test_parse_passed,
                test_intent_passed=test_intent_passed,
                test_body_system_passed=test_body_system_passed,
                test_strict_passed=test_strict_passed,
                test_mean_english_f1=test_mean_english_f1,
                baseline_eval_passed=baseline_eval_passed,
                baseline_eval_total=baseline_eval_total,
                baseline_test_total=baseline_test_total,
                baseline_test_parse_passed=baseline_test_parse_passed,
                baseline_test_intent_passed=baseline_test_intent_passed,
                baseline_test_body_system_passed=baseline_test_body_system_passed,
                baseline_test_strict_passed=baseline_test_strict_passed,
                baseline_test_mean_english_f1=baseline_test_mean_english_f1,
                train_rows=train_rows,
                dev_rows=dev_rows,
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
