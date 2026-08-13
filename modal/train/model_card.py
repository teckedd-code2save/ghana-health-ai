"""
Shared Hugging Face model-card writer for Ghana Health AI org pushes.

Every train script should call `write_and_push_model_card` after push_to_hub
so repos are never bare weight dumps.

Usage (inside a Modal GPU function with huggingface_hub installed):

    from model_card import write_and_push_model_card
    write_and_push_model_card(
        repo_id="teckedd/gha-whisper-small-twi-v6",
        task="automatic-speech-recognition",
        language=["tw", "ak"],
        base_model="openai/whisper-small",
        metrics={"wer": 0.3044, "cer": 0.1062},
        datasets=["google/WaxalNLP", "fsicoli/common_voice_22_0"],
        summary="Twi Whisper small fine-tune for Ghana Health AI.",
        extra_markdown="## Training recipe\\n...\\n",
        token=os.environ.get("HF_TOKEN"),
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def _dataset_id(value: str) -> str:
    """Return a Hub-valid dataset id, stripping local config/split annotations."""
    return (value or "").split(":", 1)[0].strip()


def _dataset_label(value: str) -> str:
    value = (value or "").strip()
    if ":" not in value:
        return value
    dataset, detail = value.split(":", 1)
    return f"{dataset} ({detail})"


def build_model_card_md(
    *,
    repo_id: str,
    task: str,
    language: list[str],
    base_model: str,
    metrics: Optional[dict[str, Any]] = None,
    datasets: Optional[list[str]] = None,
    summary: str = "",
    extra_markdown: str = "",
    license_id: str = "apache-2.0",
    tags: Optional[list[str]] = None,
    library_name: str = "transformers",
    pipeline_tag: Optional[str] = None,
) -> str:
    """YAML front-matter + human README for HF model hubs."""
    metrics = metrics or {}
    datasets = datasets or []
    tags = tags or []
    pipeline = pipeline_tag or task
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lang_yaml = "\n".join(f"- {l}" for l in language) if language else "- tw"
    dataset_ids = []
    for d in datasets:
        clean = _dataset_id(d)
        if clean and clean not in dataset_ids:
            dataset_ids.append(clean)
    ds_yaml = "\n".join(f"- {d}" for d in dataset_ids) if dataset_ids else ""
    tag_list = sorted(
        set(
            tags
            + language
            + [
                "ghana",
                "twi",
                "akan",
                "ghana-health-ai",
                "serendepify",
            ]
        )
    )
    tags_yaml = "\n".join(f"- {t}" for t in tag_list)

    metric_yaml_lines: list[str] = []
    for name, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float):
            # WER/CER as percent-friendly raw ratio in model-index
            metric_yaml_lines.append(
                f"""    - type: {name}
      value: {value:.6f}
      name: {name.upper()}"""
            )
        else:
            metric_yaml_lines.append(
                f"""    - type: {name}
      value: {value}
      name: {name}"""
            )
    metrics_block = "\n".join(metric_yaml_lines)

    model_index = ""
    if metrics_block:
        primary_dataset = dataset_ids[0] if dataset_ids else "custom"
        primary_dataset_name = _dataset_label(datasets[0]) if datasets else "custom"
        model_index = f"""
model-index:
- name: {repo_id.split("/")[-1]}
  results:
  - task:
      type: {task}
    dataset:
      type: {primary_dataset}
      name: {primary_dataset_name}
    metrics:
{metrics_block}
"""

    front = f"""---
language:
{lang_yaml}
license: {license_id}
library_name: {library_name}
pipeline_tag: {pipeline}
base_model: {base_model}
tags:
{tags_yaml}
{"datasets:" if ds_yaml else ""}
{ds_yaml}
{model_index.strip()}
---
"""

    metric_table = ""
    if metrics:
        rows = "\n".join(
            f"| `{k}` | {v:.4f} |" if isinstance(v, float) else f"| `{k}` | {v} |"
            for k, v in metrics.items()
        )
        metric_table = f"""
## Metrics

| Metric | Value |
|--------|-------|
{rows}
"""

    ds_section = ""
    if datasets:
        bullets = "\n".join(f"- `{_dataset_label(d)}`" for d in datasets)
        ds_section = f"""
## Training data

{bullets}
"""

    body = f"""
# {repo_id}

{summary or "Model trained for **Ghana Health AI** (Serendepify) — voice-first health companion for Ghana."}

- **Base model:** `{base_model}`
- **Task:** `{task}`
- **Languages:** {", ".join(f"`{l}`" for l in language) or "`tw`"}
- **Card generated:** {now} (UTC)
- **Product:** [ghanahealth.serendepify.com](https://ghanahealth.serendepify.com)

> Not a medical device. Outputs support community health guidance only.

{metric_table}
{ds_section}

## Intended use

- In-product ASR / TTS / chat for Twi (Akan) and English health conversations in Ghana.
- Research on low-resource Ghanaian language speech and health dialogue.

## Out of scope

- Clinical diagnosis or autonomous medical decisions.
- Claiming near-native quality without reporting held-out WER/CER or human A/B scores.

## How to load

```python
from transformers import pipeline  # or AutoModel + processor per task
# repo: {repo_id}
```

## Citation

If you use this checkpoint, please credit **Ghana Health AI / Serendepify** and the upstream base model authors plus any listed datasets.

{extra_markdown}
""".strip()

    # Front matter first; strip accidental double blanks
    return front.strip() + "\n\n" + body + "\n"


def write_and_push_model_card(
    repo_id: str,
    *,
    task: str,
    language: list[str],
    base_model: str,
    metrics: Optional[dict[str, Any]] = None,
    datasets: Optional[list[str]] = None,
    summary: str = "",
    extra_markdown: str = "",
    license_id: str = "apache-2.0",
    tags: Optional[list[str]] = None,
    library_name: str = "transformers",
    pipeline_tag: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = False,
) -> str:
    """
    Upload README.md (model card) to an existing or new HF repo.
    Returns the markdown that was pushed.
    """
    from huggingface_hub import HfApi

    md = build_model_card_md(
        repo_id=repo_id,
        task=task,
        language=language,
        base_model=base_model,
        metrics=metrics,
        datasets=datasets,
        summary=summary,
        extra_markdown=extra_markdown,
        license_id=license_id,
        tags=tags,
        library_name=library_name,
        pipeline_tag=pipeline_tag,
    )
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_file(
        path_or_fileobj=md.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="docs: add Ghana Health AI model card",
    )
    print(f"[model_card] pushed README → https://huggingface.co/{repo_id}")
    return md
