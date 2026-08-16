"""
Fine-tune Whisper ASR for Twi/Akan from OpenAI bases — go for gold.

Promotion bar (immutable full Waxal test n=1522, same split + decode):
  v6 greedy  WER 31.49%  CER 10.62%  ← current production, the gate
  v6 beam=5  WER 30.44%  CER 10.62%  (serving decode bar)
  Round 2 greedy WER 32.83% (historical reference only — superseded by v6)

Past failures (continued FT from Round 2 overfit val):
  v3 33.99% · v4 34.96% · v5 34.13%  — do not promote

v6 recipe (retrain FROM openai/whisper-* bases):
  Data:
    - google/WaxalNLP aka_asr train (foundation; NEVER touch test)
    - Common Voice 22 Twi (fsicoli) train + validated-only filter (CC0)
    - optional GhanaNLP Twi multispeaker (lower weight; CC BY-NC research)
  Method:
    - Full FT (encoder unfrozen) — base is English-pretrained
    - SpecAugment + speed-pert {0.9,1.0,1.1}
    - Early-stop on Waxal validation WER
    - Auto full-test gate; promote only if WER beats serving v6 (< 0.3149 greedy)
  Ladder:
    1) openai/whisper-small  → teckedd/gha-whisper-small-twi-v6
    2) openai/whisper-medium → teckedd/gha-whisper-medium-twi-v6

  modal run --detach modal/train/train_asr.py \\
    --base-model openai/whisper-small \\
    --run-name v6-small --max-steps 3000 \\
    --push-repo teckedd/gha-whisper-small-twi-v6 --no-wait

  modal run --detach modal/train/train_asr.py \\
    --base-model openai/whisper-medium \\
    --run-name v6-medium --max-steps 2500 \\
    --batch-size 4 --grad-accum 8 \\
    --push-repo teckedd/gha-whisper-medium-twi-v6 --no-wait

Secret: huggingface-token
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

APP_NAME = "ghana-health-asr-train"
DEFAULT_BASE = "openai/whisper-small"
# Serving gate: v6 is production. A new checkpoint promotes only if it beats
# v6 on the immutable full Waxal test (n=1522) under the same decode.
BASELINE_WER = 0.3149  # v6 greedy, full Waxal test n=1522
BASELINE_CER = 0.1062
BEAM5_WER = 0.3044  # v6 beam=5, full Waxal test n=1522 (serving decode)
ROUND2_WER = 0.3283  # historical reference only — superseded by v6

# Extra corpora mixed into train (Waxal always primary)
EXTRA_DATASETS = [
    {
        "name": "fsicoli/common_voice_22_0",
        "config": "tw",
        "split": "train",
        "weight": 0.25,
        "validated_only": True,
        "max_n": 5000,
        "note": "Common Voice 22 Twi CC0 — validated votes filter",
    },
    {
        "name": "fsicoli/common_voice_22_0",
        "config": "en",
        "split": "train",
        "weight": 0.20,
        "validated_only": True,
        "max_n": 5000,
        "note": "Common Voice 22 English CC0 — retention mix to reduce English regression",
    },
    {
        "name": "ghananlpcommunity/twi-speech-text-multispeaker-16k",
        "config": None,
        "split": "train",
        "weight": 0.15,
        "validated_only": False,
        "max_n": 4000,
        "note": "CC BY-NC 4.0 research train only",
    },
]

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("akan-speech-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("akan-speech-eval-results", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_LOCAL_ASR_DIR = os.path.join(_REPO_ROOT, "tmp", "asr-local-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "evaluate==0.4.3",
        "jiwer==3.0.5",
        "librosa==0.10.2.post1",
        "soundfile==0.13.1",
        "huggingface_hub==0.26.2",
        "tensorboard==2.18.0",
        "numpy<2.3",
        "tqdm",
        "pandas",
    )
    # Shared model-card helper — every hub push must ship a real README
    .add_local_file(
        local_path=os.path.join(_TRAIN_DIR, "model_card.py"),
        remote_path="/root/gha_train/model_card.py",
    )
    .add_local_dir(
        local_path=_LOCAL_ASR_DIR,
        remote_path="/root/gha_local_asr",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _normalize_text(text: str) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFC", text or "")
    t = t.lower()
    t = re.sub(r"[^\w\sɛɔáàâäéèêëíìîïóòôöúùûüńŋ]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


def _find_text_col(columns: list[str]) -> Optional[str]:
    for c in ("sentence", "text", "transcription", "transcript", "normalized_text"):
        if c in columns:
            return c
    return None


def _find_audio_col(columns: list[str]) -> str:
    if "audio" in columns:
        return "audio"
    return columns[0]


@app.function(
    image=image,
    gpu="A100",
    timeout=10 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/checkpoints": ckpt_vol,
        "/results": results_vol,
    },
    secrets=SECRETS,
    memory=65536,
)
def train(
    base_model: str = DEFAULT_BASE,
    dataset_name: str = "google/WaxalNLP",
    dataset_config: str = "aka_asr",
    max_steps: int = 3000,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    grad_accum: int = 4,
    freeze_encoder: bool = False,
    weight_decay: float = 0.01,
    waxal_weight: float = 0.60,
    use_extra_data: bool = True,
    push_repo: Optional[str] = None,
    smoke: bool = False,
    run_name: str = "v6-small",
    full_test_after: bool = True,
    train_limit: int = 0,
    eval_limit: int = 0,
    use_local_data: bool = True,
    local_manifest_path: str = "/root/gha_local_asr/manifest.jsonl",
    local_weight: float = 0.20,
) -> dict[str, Any]:
    """
    Retrain from openai/whisper-* on Waxal + Common Voice Twi (+ optional extras).
    """
    import json
    import random
    import tarfile
    from pathlib import Path

    import numpy as np
    import torch
    import evaluate
    from datasets import Audio, Dataset, interleave_datasets, load_dataset
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    if smoke:
        max_steps = min(max_steps, 20)
        batch_size = min(batch_size, 2)
        full_test_after = False
        train_limit = train_limit or 64
        eval_limit = eval_limit or 16

    # Medium needs smaller micro-batch
    if "medium" in base_model.lower() and batch_size > 4 and not smoke:
        batch_size = 4
        grad_accum = max(grad_accum, 8)
        if learning_rate > 8e-6:
            learning_rate = 8e-6

    token = _hf_token()
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    cache = "/root/.cache/huggingface"
    out_dir = f"/checkpoints/gha-asr/{run_name}_{base_model.replace('/', '_')}_s{max_steps}"
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[train] {run_name} base={base_model} steps={max_steps} "
        f"freeze_encoder={freeze_encoder} lr={learning_rate} "
        f"bs={batch_size}x{grad_accum} extra={use_extra_data} waxal_w={waxal_weight}"
    )

    def _bounded_split(split_name: str, limit: int) -> str:
        if limit and int(limit) > 0:
            return f"{split_name}[:{int(limit)}]"
        return split_name

    def _with_retries(fn, what: str, attempts: int = 4):
        """Retry transient HF CDN failures (read timeouts, 503s). The hf-cache
        volume keeps completed shards, so each retry resumes where it died."""
        import time

        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[train] {what} attempt {attempt}/{attempts} failed: {exc}",
                    flush=True,
                )
                if attempt == attempts:
                    raise
                time.sleep(15 * attempt)

    def _materialize_streaming_waxal(split_name: str, limit: int, source: str) -> Dataset:
        """
        HF split slicing still resolves every Waxal shard. For credit-safe proof runs,
        stream and materialize only the first usable examples into an in-memory Dataset.
        """
        target = int(limit or 0)
        if target <= 0:
            raise ValueError("streaming materialization requires a positive limit")

        print(f"[train] streaming {source} split={split_name} target={target}", flush=True)
        ds_stream = load_dataset(
            dataset_name,
            dataset_config,
            split=split_name,
            token=token,
            cache_dir=cache,
            streaming=True,
        )
        column_names = list(getattr(ds_stream, "column_names", []) or [])
        stream_audio_col = "audio" if "audio" in column_names else _find_audio_col(column_names)
        stream_text_col = _find_text_col(column_names)
        if stream_text_col is None:
            raise RuntimeError(f"No text column in streamed {source}: {column_names}")
        ds_stream = ds_stream.cast_column(stream_audio_col, Audio(sampling_rate=16000))

        rows: list[dict[str, Any]] = []
        skipped = 0
        for row in ds_stream:
            text = str(row.get(stream_text_col) or "")
            if len(_normalize_text(text)) < 2:
                skipped += 1
                continue
            audio = row.get(stream_audio_col)
            if not isinstance(audio, dict) or audio.get("array") is None:
                skipped += 1
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate") or 16000)
            if arr.size < int(0.25 * sr):
                skipped += 1
                continue
            rows.append(
                {
                    "audio": {"array": arr, "sampling_rate": sr},
                    "text": text,
                }
            )
            if len(rows) >= target:
                break

        if len(rows) < max(16, min(64, target // 4)):
            raise RuntimeError(
                f"Only materialized {len(rows)} {source} rows from {split_name}; skipped={skipped}"
            )
        print(f"[train] {source} materialized n={len(rows)} skipped={skipped}", flush=True)
        return Dataset.from_list(rows)

    def _materialize_streaming_extra(cfg: dict, source: str) -> Optional[Dataset]:
        """
        Materialize capped auxiliary speech data without asking HF to fetch every shard.
        This is critical for Common Voice English, where split slicing still resolves
        the full multi-tar train split before we see a single sample.
        """
        target = int(cfg.get("max_n") or 0)
        if target <= 0:
            return None

        name = cfg["name"]
        config = cfg.get("config")
        split = cfg.get("split") or "train"
        validated_only = bool(cfg.get("validated_only"))

        print(
            f"[train] streaming extra {source} dataset={name}/{config or 'default'} "
            f"split={split} target={target}",
            flush=True,
        )
        load_kwargs: dict[str, Any] = {
            "split": split,
            "token": token,
            "cache_dir": cache,
            "streaming": True,
        }
        if "common_voice" in name:
            load_kwargs["trust_remote_code"] = True
        if config:
            ds_stream = load_dataset(name, config, **load_kwargs)
        else:
            ds_stream = load_dataset(name, **load_kwargs)

        column_names = list(getattr(ds_stream, "column_names", []) or [])
        stream_audio_col = "audio" if "audio" in column_names else _find_audio_col(column_names)
        stream_text_col = _find_text_col(column_names)
        if stream_text_col is None:
            raise RuntimeError(f"No text column in streamed {source}: {column_names}")
        ds_stream = ds_stream.cast_column(stream_audio_col, Audio(sampling_rate=16000))

        rows: list[dict[str, Any]] = []
        skipped = 0
        for row in ds_stream:
            if validated_only and "up_votes" in row and "down_votes" in row:
                try:
                    if int(row.get("up_votes") or 0) < 2 or int(row.get("down_votes") or 0) != 0:
                        skipped += 1
                        continue
                except Exception:  # noqa: BLE001
                    pass

            text = str(row.get(stream_text_col) or "")
            if len(_normalize_text(text)) < 2:
                skipped += 1
                continue
            audio = row.get(stream_audio_col)
            if not isinstance(audio, dict) or audio.get("array") is None:
                skipped += 1
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate") or 16000)
            dur = arr.size / max(1, sr)
            if dur < 0.5 or dur > 28.0:
                skipped += 1
                continue

            rows.append(
                {
                    "audio": {"array": arr, "sampling_rate": sr},
                    "sentence": text,
                    "up_votes": int(row.get("up_votes") or 0) if "up_votes" in row else 0,
                    "down_votes": int(row.get("down_votes") or 0) if "down_votes" in row else 0,
                }
            )
            if len(rows) >= target:
                break

        if len(rows) < 30:
            print(f"[train] streamed extra {source} too small n={len(rows)} skipped={skipped}")
            return None
        print(f"[train] streamed extra {source} materialized n={len(rows)} skipped={skipped}")
        return Dataset.from_list(rows)

    # ── Waxal foundation ────────────────────────────────────────────
    if train_limit or eval_limit:
        waxal_train = _with_retries(
            lambda: _materialize_streaming_waxal(
                "train",
                int(train_limit or 3000),
                "waxal-train",
            ),
            "waxal-train",
        )
        try:
            eval_ds = _with_retries(
                lambda: _materialize_streaming_waxal(
                    "validation",
                    int(eval_limit or max(100, min(300, len(waxal_train) // 10))),
                    "waxal-val",
                ),
                "waxal-val",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[train] validation streaming failed ({exc}); holding out train subset")
            n = len(waxal_train)
            idx = list(range(n))
            random.Random(42).shuffle(idx)
            cut = min(max(16, int(eval_limit or max(32, n * 0.08))), max(1, n // 4))
            eval_ds = waxal_train.select(idx[:cut])
            waxal_train = waxal_train.select(idx[cut:])
        raw_test_len: int | str = "unused-streaming-proof"
        audio_col = "audio"
        text_col = "text"
    else:
        raw = load_dataset(dataset_name, dataset_config, token=token, cache_dir=cache)
        print("[train] waxal splits", list(raw.keys()))
        assert "test" in raw, "Waxal must have immutable test split"
        split_train = "train" if "train" in raw else list(raw.keys())[0]
        if "validation" in raw:
            split_eval = "validation"
        elif "dev" in raw:
            split_eval = "dev"
        else:
            split_eval = None

        waxal_train = raw[split_train]
        if split_eval:
            eval_ds = raw[split_eval]
        else:
            n = len(waxal_train)
            idx = list(range(n))
            random.Random(42).shuffle(idx)
            cut = max(1, int(0.05 * n))
            eval_ds = waxal_train.select(idx[:cut])
            waxal_train = waxal_train.select(idx[cut:])
            print(f"[train] no dev — held out {cut} from waxal train")

        raw_test_len = len(raw["test"])
        audio_col = _find_audio_col(waxal_train.column_names)
        text_col = _find_text_col(waxal_train.column_names)

    print(f"[train] waxal train={len(waxal_train)} eval={len(eval_ds)} test={raw_test_len} (unused)")

    if text_col is None:
        raise RuntimeError(f"No text column in waxal: {waxal_train.column_names}")

    def _to_audio_text(
        ds,
        a_col: str,
        t_col: str,
        source: str,
        max_n: int | None = None,
        validated_only: bool = False,
    ):
        """Filter metadata-only; rename to {audio,text}; cast 16 kHz."""
        # Common Voice vote filter (validated gold)
        if validated_only and "up_votes" in ds.column_names and "down_votes" in ds.column_names:
            before = len(ds)

            def vote_ok(ex):
                try:
                    up = int(ex.get("up_votes") or 0)
                    down = int(ex.get("down_votes") or 0)
                except Exception:  # noqa: BLE001
                    return True
                return up >= 2 and down == 0

            # filter without loading audio
            try:
                ds = ds.filter(
                    lambda up, down: int(up or 0) >= 2 and int(down or 0) == 0,
                    input_columns=["up_votes", "down_votes"],
                    desc=f"votes-{source}",
                )
            except Exception:  # noqa: BLE001
                ds = ds.filter(vote_ok, desc=f"votes-{source}")
            print(f"[train] {source} validated votes {before} → {len(ds)}")

        if "duration" in ds.column_names:
            before = len(ds)
            ds = ds.filter(
                lambda d: 0.5 <= float(d or 0) <= 28.0,
                input_columns=["duration"],
                desc=f"dur-{source}",
            )
            print(f"[train] {source} duration filter {before} → {len(ds)}")

        before = len(ds)
        ds = ds.filter(
            lambda t: len(_normalize_text(t or "")) >= 2,
            input_columns=[t_col],
            desc=f"text-{source}",
        )
        print(f"[train] {source} text filter {before} → {len(ds)}")

        if max_n and len(ds) > max_n:
            idx = list(range(len(ds)))
            random.Random(hash(source) % 10_000).shuffle(idx)
            ds = ds.select(idx[:max_n])
            print(f"[train] {source} capped to {max_n}")

        keep = [a_col, t_col]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        if a_col != "audio":
            ds = ds.rename_column(a_col, "audio")
        if t_col != "text":
            ds = ds.rename_column(t_col, "text")
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        except TypeError as exc:
            # datasets 3.1.0 + recent pyarrow: chunked cast of pre-decoded audio
            # dicts crashes ("Cannot convert ChunkedArray to Array") once the
            # table is large enough to span multiple chunks. Rebuild in small
            # parts (single-chunk casts are proven fine) so the Audio feature
            # type is preserved — interleave_datasets requires aligned features.
            sample = ds[0]["audio"] if len(ds) else None
            if not (isinstance(sample, dict) and sample.get("array") is not None):
                raise
            print(
                f"[train] {source} chunked audio-cast workaround ({exc})",
                flush=True,
            )
            from datasets import concatenate_datasets

            rows = ds.to_list()
            part_size = 800
            parts = [
                Dataset.from_list(rows[i : i + part_size]).cast_column(
                    "audio", Audio(sampling_rate=16000)
                )
                for i in range(0, len(rows), part_size)
            ]
            ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        print(f"[train] {source} ready n={len(ds)}")
        return ds

    def _split_expr(cfg: dict) -> str:
        split = cfg.get("split") or "train"
        max_n = cfg.get("max_n")
        if max_n:
            return f"{split}[:{int(max_n)}]"
        return split

    def _load_common_voice(cfg: dict) -> Optional[Dataset]:
        """Load Common Voice. Twi keeps a manual tar+tsv fallback; other configs use HF only."""
        name = cfg["name"]
        config = cfg.get("config") or "tw"
        if cfg.get("max_n"):
            streamed = _with_retries(
                lambda: _materialize_streaming_extra(cfg, f"common_voice_{config}"),
                f"common_voice_{config}",
            )
            if streamed is not None:
                return streamed
        try:
            print(f"[train] loading Common Voice {name} config={config} …")
            # Prefer train split (speaker-disjoint from test by CorporaCreator)
            eds = load_dataset(
                name,
                config,
                split=_split_expr(cfg),
                token=token,
                cache_dir=cache,
                trust_remote_code=True,
            )
            print(f"[train] CV via load_dataset n={len(eds)} cols={eds.column_names}")
            return eds
        except Exception as exc:  # noqa: BLE001
            print(f"[train] CV load_dataset failed ({exc}); trying manual tsv+tar …")

        if config != "tw":
            return None

        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd
            import soundfile as sf

            work = Path(cache) / "cv_tw_manual"
            work.mkdir(parents=True, exist_ok=True)

            # validated.tsv is the gold list; we take rows that appear in train.tsv
            # (never test) so we stay out of CV test.
            train_tsv = hf_hub_download(
                name, "transcript/tw/train.tsv", repo_type="dataset", token=token, cache_dir=cache
            )
            val_tsv = hf_hub_download(
                name, "transcript/tw/validated.tsv", repo_type="dataset", token=token, cache_dir=cache
            )
            train_tar = hf_hub_download(
                name, "audio/tw/train/tw_train_0.tar", repo_type="dataset", token=token, cache_dir=cache
            )

            train_df = pd.read_csv(train_tsv, sep="\t")
            val_df = pd.read_csv(val_tsv, sep="\t")
            # intersection: in train split AND validated
            val_paths = set(val_df["path"].astype(str))
            train_df = train_df[train_df["path"].astype(str).isin(val_paths)].copy()
            if "up_votes" in train_df.columns:
                train_df = train_df[
                    (train_df["up_votes"].fillna(0).astype(int) >= 2)
                    & (train_df["down_votes"].fillna(0).astype(int) == 0)
                ]
            print(f"[train] CV manual validated∩train rows={len(train_df)}")

            extract_dir = work / "audio_train"
            extract_dir.mkdir(exist_ok=True)
            # extract only needed mp3s if not already
            marker = extract_dir / ".extracted"
            if not marker.exists():
                with tarfile.open(train_tar, "r") as tar:
                    tar.extractall(extract_dir)
                marker.write_text("ok")

            rows = []
            for _, r in train_df.iterrows():
                path = str(r["path"])
                # files may be nested after extract
                candidates = list(extract_dir.rglob(path))
                if not candidates:
                    candidates = list(extract_dir.rglob(Path(path).name))
                if not candidates:
                    continue
                fpath = candidates[0]
                try:
                    audio_arr, sr = sf.read(str(fpath), always_2d=False)
                    if getattr(audio_arr, "ndim", 1) > 1:
                        audio_arr = audio_arr.mean(axis=1)
                    sentence = r.get("sentence") or r.get("text") or ""
                    if len(_normalize_text(str(sentence))) < 2:
                        continue
                    rows.append(
                        {
                            "audio": {"array": np.asarray(audio_arr, dtype=np.float32), "sampling_rate": int(sr)},
                            "sentence": str(sentence),
                            "up_votes": int(r.get("up_votes") or 0),
                            "down_votes": int(r.get("down_votes") or 0),
                        }
                    )
                except Exception:  # noqa: BLE001
                    continue
            print(f"[train] CV manual decoded clips={len(rows)}")
            if len(rows) < 20:
                return None
            return Dataset.from_list(rows)
        except Exception as exc2:  # noqa: BLE001
            print(f"[train] CV manual load failed: {exc2}")
            return None

    waxal_std = _to_audio_text(waxal_train, audio_col, text_col, "waxal")
    eval_std = _to_audio_text(eval_ds, audio_col, text_col, "waxal-val")

    train_parts: list[Dataset] = [waxal_std]
    mix_weights: list[float] = [waxal_weight]
    sources_used = [f"{dataset_name}:{dataset_config}"]

    def _load_local_manifest(manifest_path: str) -> Optional[Dataset]:
        if not use_local_data:
            return None
        if not manifest_path or not os.path.exists(manifest_path):
            print(f"[train] local ASR manifest not found: {manifest_path}")
            return None

        local_rows: list[dict[str, Any]] = []
        skipped = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                audio_path = str(item.get("audio_path") or "")
                text = str(item.get("reference") or item.get("text") or "")
                if not audio_path or not os.path.exists(audio_path):
                    skipped += 1
                    continue
                if len(_normalize_text(text)) < 2:
                    skipped += 1
                    continue
                local_rows.append({"audio": audio_path, "text": text})

        if len(local_rows) < 8:
            print(
                f"[train] local ASR manifest too small n={len(local_rows)} skipped={skipped}"
            )
            return None

        ds = Dataset.from_list(local_rows).cast_column(
            "audio", Audio(sampling_rate=16000)
        )
        print(
            f"[train] local ASR ready n={len(ds)} skipped={skipped} manifest={manifest_path}"
        )
        return ds

    local_std = _load_local_manifest(local_manifest_path)
    if local_std is not None:
        train_parts.append(local_std)
        mix_weights.append(max(0.01, float(local_weight)))
        sources_used.append("local:ghana-health-ai-recorder")

    if use_extra_data and not smoke:
        for extra in EXTRA_DATASETS:
            extra = dict(extra)
            if train_limit:
                scaled_max = int(
                    max(
                        64,
                        min(
                            int(extra.get("max_n") or train_limit),
                            round(int(train_limit) * float(extra.get("weight") or 0.2) / max(waxal_weight, 0.05)),
                        ),
                    )
                )
                extra["max_n"] = scaled_max
            try:
                if "common_voice" in extra["name"]:
                    eds = _load_common_voice(extra)
                    if eds is None:
                        continue
                else:
                    print(f"[train] loading extra {extra['name']} …")
                    load_kwargs: dict[str, Any] = {
                        "token": token,
                        "cache_dir": cache,
                        "split": _split_expr(extra),
                    }
                    if extra.get("config"):
                        eds = load_dataset(extra["name"], extra["config"], **load_kwargs)
                    else:
                        eds = load_dataset(extra["name"], **load_kwargs)
                    if hasattr(eds, "keys") and not hasattr(eds, "column_names"):
                        eds = eds[extra.get("split") or "train"]

                ea = _find_audio_col(eds.column_names)
                et = _find_text_col(eds.column_names)
                if et is None:
                    print(f"[train] skip {extra['name']}: no text {eds.column_names}")
                    continue
                std = _to_audio_text(
                    eds,
                    ea,
                    et,
                    extra["name"].split("/")[-1],
                    max_n=extra.get("max_n"),
                    validated_only=bool(extra.get("validated_only")),
                )
                if len(std) < 30:
                    print("[train] skip extra — too small")
                    continue
                train_parts.append(std)
                mix_weights.append(float(extra.get("weight") or 0.2))
                sources_used.append(
                    f"{extra['name']}:{extra.get('config') or 'default'}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[train] extra failed {extra['name']}: {exc}")

    wsum = sum(mix_weights)
    mix_weights = [w / wsum for w in mix_weights]

    if len(train_parts) == 1:
        train_std = train_parts[0]
        print(f"[train] single-source n={len(train_std)}")
    else:
        train_std = interleave_datasets(
            train_parts,
            probabilities=mix_weights,
            seed=42,
            stopping_strategy="all_exhausted",
        )
        print(
            f"[train] MIX sources={sources_used} weights={mix_weights} "
            f"parts={[len(p) for p in train_parts]}"
        )

    if smoke:
        train_std = train_std.select(range(min(32, len(train_std))))
        eval_std = eval_std.select(range(min(16, len(eval_std))))

    processor = WhisperProcessor.from_pretrained(base_model, cache_dir=cache, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model, cache_dir=cache, token=token
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.forced_decoder_ids = None
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.num_beams = 1
    model.generation_config.max_length = 225
    print(
        f"[train] decoder_start={model.config.decoder_start_token_id} "
        f"pad={model.config.pad_token_id}"
    )

    if freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable {trainable:,}/{total:,} freeze_encoder={freeze_encoder}")

    SPEEDS = (0.9, 1.0, 1.1)

    def prepare_train(batch: dict) -> dict:
        audio = batch["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or 16000)
        dur = len(arr) / float(sr)
        if dur < 0.3:
            pad = max(0, int(0.5 * sr) - len(arr))
            if pad:
                arr = np.concatenate([arr, np.zeros(pad, dtype=np.float32)])
        elif dur > 30.0:
            arr = arr[: int(30 * sr)]
        speed = random.choice(SPEEDS)
        if speed != 1.0 and len(arr) > 160:
            n_out = max(1, int(len(arr) / speed))
            x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            arr = np.interp(x_new, x_old, arr).astype(np.float32)
        feats = processor.feature_extractor(
            arr, sampling_rate=sr, return_tensors="np"
        ).input_features[0]
        text = _normalize_text(batch["text"]) or " "
        labels = processor.tokenizer(text).input_ids
        if not labels:
            labels = [processor.tokenizer.eos_token_id or 50256]
        return {"input_features": feats, "labels": labels}

    def prepare_eval(batch: dict) -> dict:
        audio = batch["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate") or 16000)
        if len(arr) < int(0.3 * sr):
            arr = np.concatenate(
                [arr, np.zeros(max(0, int(0.5 * sr) - len(arr)), dtype=np.float32)]
            )
        elif len(arr) > int(30 * sr):
            arr = arr[: int(30 * sr)]
        feats = processor.feature_extractor(
            arr, sampling_rate=sr, return_tensors="np"
        ).input_features[0]
        text = _normalize_text(batch["text"]) or " "
        labels = processor.tokenizer(text).input_ids
        if not labels:
            labels = [processor.tokenizer.eos_token_id or 50256]
        return {"input_features": feats, "labels": labels}

    train_prep = train_std.map(
        prepare_train,
        remove_columns=train_std.column_names,
        desc="prep-train",
        writer_batch_size=64,
    )
    eval_prep = eval_std.map(
        prepare_eval,
        remove_columns=eval_std.column_names,
        desc="prep-eval",
        writer_batch_size=64,
    )

    class DataCollatorSpecAug:
        def __init__(self, processor, train: bool = True, strong: bool = True):
            self.processor = processor
            self.train = train
            self.mask_time_prob = 0.10 if strong else 0.05
            self.mask_time_length = 12 if strong else 10
            self.mask_feature_prob = 0.08 if strong else 0.05
            self.mask_feature_length = 20 if strong else 16

        def _spec_augment(self, feats: torch.Tensor) -> torch.Tensor:
            if not self.train or feats.dim() != 3:
                return feats
            b, n_mels, t = feats.shape
            out = feats.clone()
            for i in range(b):
                num_t = max(1, int(self.mask_time_prob * t / max(1, self.mask_time_length)))
                for _ in range(num_t):
                    length = random.randint(1, self.mask_time_length)
                    if t - length <= 0:
                        continue
                    start = random.randint(0, t - length)
                    out[i, :, start : start + length] = 0
                num_f = max(
                    1, int(self.mask_feature_prob * n_mels / max(1, self.mask_feature_length))
                )
                for _ in range(num_f):
                    length = random.randint(1, self.mask_feature_length)
                    if n_mels - length <= 0:
                        continue
                    start = random.randint(0, n_mels - length)
                    out[i, start : start + length, :] = 0
            return out

        def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
            clean = []
            for f in features:
                labs = f.get("labels") or []
                if isinstance(labs, torch.Tensor):
                    labs = labs.tolist()
                if labs:
                    clean.append({"input_features": f["input_features"], "labels": labs})
            if not clean:
                f0 = features[0]
                clean = [
                    {
                        "input_features": f0["input_features"],
                        "labels": [self.processor.tokenizer.eos_token_id or 50256],
                    }
                ]
            input_features = [{"input_features": f["input_features"]} for f in clean]
            label_features = [{"input_ids": f["labels"]} for f in clean]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (
                self.processor.tokenizer.bos_token_id is not None
                and labels.size(1) > 1
                and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
            ):
                labels = labels[:, 1:]
            for i in range(labels.size(0)):
                if (labels[i] != -100).sum().item() == 0:
                    labels[i, 0] = self.processor.tokenizer.eos_token_id or 50256
            batch["input_features"] = self._spec_augment(batch["input_features"])
            batch["labels"] = labels
            return batch

    train_collator = DataCollatorSpecAug(processor, train=True, strong=True)
    eval_collator = DataCollatorSpecAug(processor, train=False, strong=False)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [_normalize_text(s) for s in pred_str]
        label_str = [_normalize_text(s) for s in label_str]
        return {
            "wer": float(wer_metric.compute(predictions=pred_str, references=label_str)),
            "cer": float(cer_metric.compute(predictions=pred_str, references=label_str)),
        }

    eval_steps = 100 if not smoke else 10
    if "medium" in base_model.lower():
        eval_steps = 80 if not smoke else 10

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=min(200, max(20, max_steps // 10)),
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        # Never use label_smoothing with Whisper — Trainer pops labels before forward
        label_smoothing_factor=0.0,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=not smoke,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        max_grad_norm=1.0,
        gradient_checkpointing=bool("medium" in base_model.lower()),
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_prep,
        eval_dataset=eval_prep,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=(
            []
            if smoke
            else [
                EarlyStoppingCallback(
                    early_stopping_patience=5,
                    early_stopping_threshold=0.002,
                )
            ]
        ),
    )

    _orig_get_eval = trainer.get_eval_dataloader

    def get_eval_dataloader(eval_dataset=None):
        old = trainer.data_collator
        trainer.data_collator = eval_collator
        try:
            return _orig_get_eval(eval_dataset)
        finally:
            trainer.data_collator = old

    trainer.get_eval_dataloader = get_eval_dataloader  # type: ignore[method-assign]

    train_result = trainer.train()
    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()

    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)
    ckpt_vol.commit()

    val_wer = float(eval_metrics.get("eval_wer", 1.0))
    val_cer = float(eval_metrics.get("eval_cer", 1.0))

    full_test: Optional[dict[str, Any]] = None
    if full_test_after and not smoke:
        print("[train] FULL Waxal test (promotion gate, greedy) …")
        full_test = _run_full_test(
            model=model,
            processor=processor,
            token=token,
            cache=cache,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_beams=1,
        )
        print("[train] full test greedy", full_test)
        # Also beam=5 for fair compare to production serving
        print("[train] FULL Waxal test beam=5 …")
        full_test_beam = _run_full_test(
            model=model,
            processor=processor,
            token=token,
            cache=cache,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_beams=5,
        )
        print("[train] full test beam5", full_test_beam)
        full_test["beam5"] = full_test_beam

    beats_greedy = bool(full_test and full_test["wer"] < BASELINE_WER)
    beats_beam = bool(
        full_test
        and full_test.get("beam5")
        and full_test["beam5"]["wer"] < BEAM5_WER
    )

    hub_status = None
    if push_repo and not smoke and token:
        try:
            model.push_to_hub(push_repo, token=token, private=False)
            processor.push_to_hub(push_repo, token=token, private=False)
            # Proper model card — never leave a bare weight dump on the org
            import sys

            sys.path.insert(0, "/root/gha_train")
            from model_card import write_and_push_model_card  # type: ignore

            card_metrics: dict[str, Any] = {
                "val_wer": val_wer,
                "val_cer": val_cer,
            }
            if full_test:
                card_metrics["wer"] = full_test.get("wer")
                card_metrics["cer"] = full_test.get("cer")
                if full_test.get("beam5"):
                    card_metrics["wer_beam5"] = full_test["beam5"].get("wer")
                    card_metrics["cer_beam5"] = full_test["beam5"].get("cer")

            write_and_push_model_card(
                push_repo,
                task="automatic-speech-recognition",
                language=(
                    ["tw", "ak", "en"]
                    if any(":en" in s for s in sources_used)
                    else ["tw", "ak"]
                ),
                base_model=base_model,
                metrics=card_metrics,
                datasets=[s for s in (sources_used or []) if s]
                or ["google/WaxalNLP", "fsicoli/common_voice_22_0"],
                summary=(
                    f"Twi/Akan"
                    f"{' + English retention' if any(':en' in s for s in sources_used) else ''} "
                    f"Whisper ASR for Ghana Health AI (`{run_name}`). "
                    f"Recipe v6 mix; promote={beats_greedy}."
                ),
                extra_markdown=f"""
## Recipe

- Run: `{run_name}`
- Freeze encoder: `{freeze_encoder}`
- LR / steps / batch: `{learning_rate}` / `{max_steps}` / `{batch_size}x{grad_accum}`
- Local recorder data: `{use_local_data}` · manifest `{local_manifest_path}` · weight `{local_weight}`
- Baseline to beat (v6 serving, Waxal full test greedy): `{BASELINE_WER}`
- Beats v6 greedy: **{beats_greedy}** · v6 beam5 bar (`{BEAM5_WER}`): **{beats_beam}**
- Historical reference (Round 2 greedy, superseded): `{ROUND2_WER}`

## Serving

Production decode uses `num_beams=5` in `modal/asr_service.py`.
""",
                tags=["whisper", "asr", "speech-recognition"],
                pipeline_tag="automatic-speech-recognition",
                token=token,
            )
            hub_status = f"pushed:{push_repo}+card"
            print(f"[train] hub push ok → {push_repo}")
        except Exception as exc:  # noqa: BLE001
            hub_status = f"push_failed:{exc}"
            print(f"[train] hub push failed: {exc}")

    summary = {
        "status": "ok",
        "run_name": run_name,
        "base_model": base_model,
        "output_dir": out_dir,
        "recipe": "v6-from-openai-mix-cv-waxal",
        "sources": sources_used,
        "mix_weights": mix_weights,
        "freeze_encoder": freeze_encoder,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "baseline_wer_to_beat": BASELINE_WER,
        "beam5_bar": BEAM5_WER,
        "val_wer": val_wer,
        "val_cer": val_cer,
        "full_test": full_test,
        "beats_v6_greedy": beats_greedy,
        "beats_v6_beam5": beats_beam,
        "round2_reference_wer": ROUND2_WER,
        "promote": beats_greedy,
        "train_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()
        },
        "eval_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_metrics.items()
        },
        "hub": hub_status,
        "push_repo": push_repo,
        "note": (
            "PROMOTE" if beats_greedy else "DO NOT PROMOTE — keep v6 serving"
        ),
    }
    with open(f"/results/train_{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    print("[train] done", summary)
    return summary


def _run_full_test(model, processor, token, cache, device, num_beams: int = 1) -> dict[str, Any]:
    import evaluate
    import time
    import torch
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    raw = None
    for attempt in range(1, 5):
        try:
            raw = load_dataset("google/WaxalNLP", "aka_asr", token=token, cache_dir=cache)
            break
        except Exception as exc:  # noqa: BLE001 — transient HF CDN failures
            print(f"[train] full-test load attempt {attempt}/4 failed: {exc}", flush=True)
            if attempt == 4:
                raise
            time.sleep(15 * attempt)
    split = "test" if "test" in raw else list(raw.keys())[-1]
    ds = raw[split]
    audio_col = "audio" if "audio" in ds.column_names else ds.column_names[0]
    text_col = _find_text_col(ds.column_names)
    assert text_col
    ds = ds.cast_column(audio_col, Audio(sampling_rate=16000))

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")
    preds: list[str] = []
    refs: list[str] = []
    model.eval()

    for row in tqdm(ds, desc=f"full-test-b{num_beams}"):
        audio = row[audio_col]
        inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            ids = model.generate(
                input_features,
                max_new_tokens=225,
                num_beams=max(1, int(num_beams)),
            )
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        preds.append(_normalize_text(hyp))
        refs.append(_normalize_text(row[text_col]))

    wer = float(wer_m.compute(predictions=preds, references=refs))
    cer = float(cer_m.compute(predictions=preds, references=refs))
    return {
        "dataset": f"google/WaxalNLP/aka_asr:{split}",
        "n": len(preds),
        "num_beams": int(num_beams),
        "wer": wer,
        "cer": cer,
        "wer_pct": round(wer * 100, 2),
        "cer_pct": round(cer * 100, 2),
        "baseline_wer_pct": round(BASELINE_WER * 100, 2),
        "delta_wer_pp": round((wer - BASELINE_WER) * 100, 2),
    }


@app.local_entrypoint()
def main(
    base_model: str = DEFAULT_BASE,
    max_steps: int = 3000,
    push_repo: str = "teckedd/gha-whisper-small-twi-v6",
    smoke: bool = False,
    freeze_encoder: bool = False,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    grad_accum: int = 4,
    run_name: str = "v6-small",
    use_extra_data: bool = True,
    waxal_weight: float = 0.60,
    full_test_after: bool = True,
    train_limit: int = 0,
    eval_limit: int = 0,
    use_local_data: bool = True,
    local_manifest_path: str = "/root/gha_local_asr/manifest.jsonl",
    local_weight: float = 0.20,
    wait: bool = True,
):
    """
    spawn() so `modal run --detach` keeps the GPU job alive.
    --no-wait: fire and forget.
    """
    call = train.spawn(
        base_model=base_model,
        max_steps=max_steps,
        push_repo=push_repo or None,
        smoke=smoke,
        freeze_encoder=freeze_encoder,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        run_name=run_name,
        use_extra_data=use_extra_data,
        waxal_weight=waxal_weight,
        full_test_after=full_test_after,
        train_limit=train_limit,
        eval_limit=eval_limit,
        use_local_data=use_local_data,
        local_manifest_path=local_manifest_path,
        local_weight=local_weight,
    )
    print(f"[train] spawned {call.object_id} run={run_name} base={base_model}")
    print("[train] follow Modal dashboard; summary → akan-speech-eval-results")
    if not wait:
        print("[train] --no-wait: exiting; job continues on Modal")
        return

    result = call.get()
    print(result)
    if result.get("full_test"):
        ft = result["full_test"]
        b5 = ft.get("beam5") or {}
        print(
            f"\n=== PROMOTION GATE ===\n"
            f"  v6 (serving) greedy: WER {BASELINE_WER*100:.2f}%\n"
            f"  v6 (serving) beam5:  WER {BEAM5_WER*100:.2f}%\n"
            f"  Round 2 (ref) greedy: WER {ROUND2_WER*100:.2f}%\n"
            f"  {run_name} greedy: WER {ft.get('wer_pct')}%  CER {ft.get('cer_pct')}% "
            f"(Δ {ft.get('delta_wer_pp'):+.2f} pp)\n"
            f"  {run_name} beam5:  WER {b5.get('wer_pct', 'n/a')}%\n"
            f"  → {'PROMOTE' if result.get('beats_v6_greedy') else 'KEEP V6 SERVING'}\n"
        )
