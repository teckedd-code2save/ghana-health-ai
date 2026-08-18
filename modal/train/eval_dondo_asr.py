"""
Evaluate KhayaAI DONDO / w2v-BERT CTC ASR on the same promotion sets
we use for Ghana Health AI.

This is the first step before spending credits on DONDO fine-tuning:
measure zero-shot / base behavior against v6 Whisper on Twi, English,
health phrases, and browser audio.

Examples:

  modal run modal/train/eval_dondo_asr.py \\
    --model-id KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en \\
    --dataset-name google/WaxalNLP --dataset-config aka_asr \\
    --split test --language "Asante Twi" --max-samples 500

  modal run modal/train/eval_dondo_asr.py \\
    --dataset-name mozilla-foundation/common_voice_17_0 \\
    --dataset-config en --split validation --language "African English" \\
    --max-samples 500
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

app = modal.App("ghana-health-dondo-asr-eval")
hf_cache = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("akan-speech-eval-results", create_if_missing=True)
lm_vol = modal.Volume.from_name("akan-speech-lm", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_LOCAL_ASR_DIR = os.path.join(_REPO_ROOT, "tmp", "asr-local-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "g++")
    # cmake <4 pinned early: pip's isolated build env would pull cmake 4.x,
    # which rejects kenlm 0.2.0's CMakeLists (see train_dondo_asr.py).
    .pip_install("cmake==3.31.6")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "evaluate==0.4.3",
        "jiwer==3.0.5",
        "librosa==0.10.2.post1",
        "numpy<2.3",
        "tqdm",
        "soundfile==0.13.1",
        "huggingface_hub==0.26.2",
        "pyctcdecode==0.5.0",
    )
    .run_commands("pip install --no-build-isolation kenlm==0.2.0")
    .add_local_dir(
        local_path=_LOCAL_ASR_DIR,
        remote_path="/root/gha_local_asr",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


LANGUAGE_MAP = {
    "Adangme": 0,
    "Akuapem Twi": 1,
    "Asante Twi": 2,
    "Dagbani": 3,
    "Dagaare": 4,
    "Ewe": 5,
    "African English": 6,
    "Fante": 7,
    "French": 8,
    "Ga": 9,
    "Gonja": 10,
    "Gurene": 11,
    "Hausa": 12,
    "Igbo": 13,
    "Kasem": 14,
    "Kikuyu": 15,
    "Konkomba (Likpakpaanl)": 16,
    "Konkomba (Likoonli)": 17,
    "Krio": 18,
    "Kusaal": 19,
    "Luo": 20,
    "Mampruli": 21,
    "Mende": 22,
    "Meru/Kimeru": 23,
    "Nzema": 24,
    "Pidgin": 25,
    "Shona": 26,
    "Swahili": 27,
    "Temne": 28,
    "Wali": 29,
    "Wolof": 30,
    "Yoruba": 31,
}


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


def _normalize_text(text: str) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFC", text or "").lower()
    t = re.sub(r"[^\w\sɛɔáàâäéèêëíìîïóòôöúùûüńŋ']", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _find_text_col(columns: list[str]) -> Optional[str]:
    for col in ("sentence", "text", "transcription", "transcript", "normalized_text"):
        if col in columns:
            return col
    return None


def _find_audio_col(columns: list[str]) -> Optional[str]:
    if "audio" in columns:
        return "audio"
    for col in columns:
        if "audio" in col.lower():
            return col
    return None


def _add_language_prefix(features, lang_id: int):
    import torch

    if features.dim() == 3:
        features = features.squeeze(0)
    time, dim = features.shape
    del time
    lang_vec = torch.zeros(dim, dtype=features.dtype, device=features.device)
    lang_vec[lang_id % dim] = 1.0
    return torch.cat([lang_vec.unsqueeze(0), features], dim=0).unsqueeze(0)


@app.function(
    image=image,
    gpu="T4",
    timeout=3 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results_vol,
        "/lm": lm_vol,
    },
    secrets=SECRETS,
)
def evaluate_dondo(
    model_id: str = "KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en",
    dataset_name: str = "google/WaxalNLP",
    dataset_config: Optional[str] = "aka_asr",
    split: str = "test",
    language: str = "Asante Twi",
    max_samples: int = 500,
    streaming: bool = True,
    local_manifest_path: Optional[str] = None,
    lm_path: str = "",
) -> dict[str, Any]:
    import json
    import torch
    import evaluate
    from datasets import Audio, load_dataset
    from transformers import AutoModelForCTC, AutoProcessor
    from tqdm import tqdm

    token = _hf_token()
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lang_id = LANGUAGE_MAP.get(language)
    if lang_id is None:
        raise RuntimeError(f"Unknown DONDO language '{language}'. Known: {sorted(LANGUAGE_MAP)}")

    print(f"[eval-dondo] load processor {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache, token=token)
    print(f"[eval-dondo] load model {model_id}", flush=True)
    model = AutoModelForCTC.from_pretrained(model_id, cache_dir=cache, token=token).to(device)
    model.eval()
    print(f"[eval-dondo] model ready device={device}", flush=True)

    if local_manifest_path:
        # Local recorder corpus mode: manifest.jsonl rows with
        # audio_path / reference / bucket (mounted at /root/gha_local_asr).
        import soundfile as sf

        if not os.path.exists(local_manifest_path):
            raise RuntimeError(f"Local manifest not found: {local_manifest_path}")
        local_rows: list[dict[str, Any]] = []
        skipped = 0
        with open(local_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                ap = str(item.get("audio_path") or "")
                text = str(item.get("reference") or item.get("text") or "")
                if not ap or not os.path.exists(ap) or len(_normalize_text(text)) < 2:
                    skipped += 1
                    continue
                arr, sr = sf.read(ap, dtype="float32")
                if getattr(arr, "ndim", 1) > 1:
                    arr = arr.mean(axis=1)
                if sr != 16000:
                    import librosa

                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                local_rows.append(
                    {
                        "array": arr,
                        "text": text,
                        "bucket": str(item.get("bucket") or "unknown"),
                    }
                )
        if max_samples and len(local_rows) > max_samples:
            local_rows = local_rows[:max_samples]
        print(
            f"[eval-dondo] local manifest ready n={len(local_rows)} skipped={skipped} "
            f"manifest={local_manifest_path}",
            flush=True,
        )

        def sample_rows():
            yield from local_rows

    else:
        print(
            f"[eval-dondo] load dataset {dataset_name}/{dataset_config or ''}:{split} streaming={streaming}",
            flush=True,
        )
        if streaming:
            if dataset_config:
                ds = load_dataset(
                    dataset_name,
                    dataset_config,
                    split=split,
                    token=token,
                    cache_dir=cache,
                    streaming=True,
                )
            else:
                ds = load_dataset(
                    dataset_name,
                    split=split,
                    token=token,
                    cache_dir=cache,
                    streaming=True,
                )
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            column_names = list(getattr(ds, "column_names", []) or [])
            sample_iter = iter(ds.take(max_samples) if max_samples else ds)
        else:
            if dataset_config:
                raw = load_dataset(dataset_name, dataset_config, token=token, cache_dir=cache)
            else:
                raw = load_dataset(dataset_name, token=token, cache_dir=cache)
            if split not in raw:
                split = list(raw.keys())[-1]
            ds = raw[split]
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            sample_iter = iter(ds)
            column_names = list(ds.column_names)
        print(f"[eval-dondo] dataset ready cols={column_names}", flush=True)

        import itertools

        first_row = next(sample_iter, None)
        if first_row is None:
            raise RuntimeError("Dataset produced no rows")
        if not column_names:
            column_names = list(first_row.keys())
        audio_col = _find_audio_col(column_names)
        text_col = _find_text_col(column_names)
        if audio_col is None:
            raise RuntimeError(f"No audio column in dataset: {column_names}")
        if text_col is None:
            raise RuntimeError(f"No transcript column in dataset: {column_names}")

        def sample_rows():
            for row in itertools.chain([first_row], sample_iter):
                yield {
                    "array": row[audio_col]["array"],
                    "text": str(row[text_col]),
                    "bucket": f"{dataset_config or 'default'}:{split}",
                }

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")
    preds: list[str] = []
    refs: list[str] = []
    bucket_preds: dict[str, list[str]] = {}
    bucket_refs: dict[str, list[str]] = {}

    decoder = None
    if lm_path:
        if not os.path.exists(lm_path):
            print(f"[eval-dondo] lm file missing: {lm_path}; greedy-only", flush=True)
        else:
            try:
                import kenlm  # noqa: F401
                from pyctcdecode import build_ctcdecoder

                vocab = processor.tokenizer.get_vocab()
                sorted_tokens = [
                    tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])
                ]
                decoder = build_ctcdecoder(sorted_tokens, kenlm_model_path=lm_path)
                print(f"[eval-dondo] beam+LM decode enabled lm={lm_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[eval-dondo] LM decode unavailable ({exc}); greedy-only", flush=True)
    lm_preds: list[str] = []
    lm_bucket_preds: dict[str, list[str]] = {}

    for row in tqdm(sample_rows(), desc=f"dondo-eval {language}"):
        proc = processor(row["array"], sampling_rate=16000, return_tensors="pt")
        feats = getattr(proc, "input_features", None)
        if feats is None:
            values = getattr(proc, "input_values", None)
            if values is None:
                raise RuntimeError("Processor returned neither input_features nor input_values")
            feats = values
        feats = _add_language_prefix(feats.to(device), lang_id)
        with torch.no_grad():
            logits = model(input_features=feats).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(pred_ids)[0]
        pred = _normalize_text(hyp)
        ref = _normalize_text(row["text"])
        preds.append(pred)
        refs.append(ref)
        bucket_preds.setdefault(row["bucket"], []).append(pred)
        bucket_refs.setdefault(row["bucket"], []).append(ref)
        if decoder is not None:
            lm_hyp = " ".join(
                str(decoder.decode(logits[0].detach().cpu().numpy())).split("|")
            )
            lm_pred = _normalize_text(lm_hyp)
            lm_preds.append(lm_pred)
            lm_bucket_preds.setdefault(row["bucket"], []).append(lm_pred)

    wer = float(wer_m.compute(predictions=preds, references=refs))
    cer = float(cer_m.compute(predictions=preds, references=refs))
    per_bucket = {
        b: {
            "n": len(bp),
            "wer_pct": round(
                float(wer_m.compute(predictions=bp, references=bucket_refs[b])) * 100, 2
            ),
            "cer_pct": round(
                float(cer_m.compute(predictions=bp, references=bucket_refs[b])) * 100, 2
            ),
        }
        for b, bp in sorted(bucket_preds.items())
    }
    lm_decode = None
    if decoder is not None and lm_preds:
        lm_decode = {
            "lm_path": lm_path,
            "wer_pct": round(
                float(wer_m.compute(predictions=lm_preds, references=refs)) * 100, 2
            ),
            "cer_pct": round(
                float(cer_m.compute(predictions=lm_preds, references=refs)) * 100, 2
            ),
            "per_bucket": {
                b: {
                    "n": len(bp),
                    "wer_pct": round(
                        float(wer_m.compute(predictions=bp, references=bucket_refs[b]))
                        * 100,
                        2,
                    ),
                    "cer_pct": round(
                        float(cer_m.compute(predictions=bp, references=bucket_refs[b]))
                        * 100,
                        2,
                    ),
                }
                for b, bp in sorted(lm_bucket_preds.items())
            },
        }
    result = {
        "model_id": model_id,
        "architecture": "wav2vec2-bert-ctc",
        "dataset": (
            f"local:{local_manifest_path}"
            if local_manifest_path
            else f"{dataset_name}/{dataset_config or ''}:{split}"
        ),
        "local_manifest": local_manifest_path,
        "per_bucket": per_bucket,
        "lm_decode": lm_decode,
        "language": language,
        "language_id": lang_id,
        "n": len(preds),
        "streaming": bool(streaming and not local_manifest_path),
        "wer": wer,
        "cer": cer,
        "wer_pct": round(wer * 100, 2),
        "cer_pct": round(cer * 100, 2),
        "promotion_note": (
            "Use this as a product gate against teckedd/gha-whisper-small-twi-v6. "
            "Do not promote without English retention and health-domain checks."
        ),
    }

    safe_name = (
        f"dondo_{model_id.replace('/', '_')}__"
        + (
            "local_recorder_corpus__manifest"
            if local_manifest_path
            else f"{dataset_name.replace('/', '_')}__{dataset_config or 'default'}__{split}"
        )
        + f"__{language.replace(' ', '_')}"
        f"_n{len(preds)}{'_streaming' if streaming and not local_manifest_path else ''}"
    )
    out_path = f"/results/{safe_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(result)
    print(f"[eval-dondo] wrote {out_path}")
    return result


@app.local_entrypoint()
def main(
    model_id: str = "KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en",
    dataset_name: str = "google/WaxalNLP",
    dataset_config: str = "aka_asr",
    split: str = "test",
    language: str = "Asante Twi",
    max_samples: int = 500,
    streaming: bool = True,
    local_manifest_path: str = "",
    lm_path: str = "",
    wait: bool = True,
):
    call = evaluate_dondo.spawn(
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_config=dataset_config or None,
        split=split,
        language=language,
        max_samples=max_samples,
        streaming=streaming,
        lm_path=lm_path,
        local_manifest_path=local_manifest_path or None,
    )
    print(f"[eval-dondo] spawned {call.object_id} model={model_id} lang={language}")
    if wait:
        print(call.get())
