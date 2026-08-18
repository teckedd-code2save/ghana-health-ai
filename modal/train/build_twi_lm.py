"""
Build a Twi 3-gram KenLM language model for CTC beam-search decoding.

The resulting /lm/twi_3gram.arpa and /lm/twi_3gram.bin are consumed by
modal/train/train_dondo_asr.py's --lm-path eval option (pyctcdecode
BeamSearchDecoderCTC).

Run (foreground):

  modal run modal/train/build_twi_lm.py

Corpus sources:
- google/WaxalNLP aka_asr train split, streamed (cap 20000 transcripts)
- local recorder manifests (/root/gha_local_asr/manifest.jsonl and
  manifest.train32.jsonl), "reference" strings

Text is normalized with the same _normalize_text as train_dondo_asr.py so the
LM token distribution matches training labels.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

APP_NAME = "ghana-health-twi-lm"
WAXAL_CAP = 20000  # same streaming cap as train_dondo_asr.py's WAXAL_FULL_CAP

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("akan-speech-hf-cache", create_if_missing=True)
lm_vol = modal.Volume.from_name("akan-speech-lm", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TRAIN_DIR))
_LOCAL_ASR_DIR = os.path.join(_REPO_ROOT, "tmp", "asr-local-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # g++ compiles kenlm from sdist (no manylinux wheels). cmake must be <4:
    # cmake 4.x rejects kenlm 0.2.0's ancient CMakeLists. Installed as a
    # separate early layer so the binary is on PATH when kenlm builds.
    # git/make are needed to clone kenlm and build the lmplz/build_binary
    # binaries (the pip package ships only the python bindings).
    # Boost (program_options/system/thread/test) is required by the kenlm
    # source build of the lmplz / build_binary CLI tools.
    .apt_install(
        "g++",
        "git",
        "make",
        # libsndfile1 backs soundfile, needed to stream Waxal rows (audio col)
        "libsndfile1",
        "libboost-program-options-dev",
        "libboost-system-dev",
        "libboost-thread-dev",
        "libboost-test-dev",
    )
    .pip_install("cmake==3.31.6")
    .pip_install(
        "datasets==3.1.0",
        "huggingface_hub==0.26.2",
        # Waxal's audio column decodes on iteration even when only reading text
        "soundfile==0.13.1",
    )
    # kenlm must build WITHOUT pip's isolated build env (which would pull the
    # latest cmake 4.x and fail on kenlm's ancient CMakeLists); the pinned
    # cmake 3.31.6 layer above is used instead.
    .run_commands("pip install --no-build-isolation kenlm==0.2.0")
    # Build the lmplz / build_binary CLI tools from the kenlm source tree.
    .run_commands(
        "cd /tmp && git clone --depth 1 https://github.com/kpu/kenlm.git "
        "&& cd kenlm && mkdir -p build && cd build "
        "&& cmake .. -DCMAKE_BUILD_TYPE=Release "
        "&& make -j4 lmplz build_binary"
    )
    # Local recorder corpus (manifest.jsonl / manifest.train32.jsonl + audio/)
    .add_local_dir(
        local_path=_LOCAL_ASR_DIR,
        remote_path="/root/gha_local_asr",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )


def _normalize_text(text: str) -> str:
    """Same normalization as train_dondo_asr.py: lowercase, NFC, strip
    punctuation except Twi letters."""
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


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/lm": lm_vol,
    },
    secrets=SECRETS,
)
def build_twi_lm(waxal_limit: int = WAXAL_CAP) -> dict[str, Any]:
    import json
    import subprocess

    token = _hf_token()
    cache = "/root/.cache/huggingface"
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    def _with_retries(fn, what: str, attempts: int = 4):
        """Retry transient HF CDN failures (read timeouts, 503s). The hf-cache
        volume keeps completed shards, so each retry resumes where it died."""
        import time

        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[twi-lm] {what} attempt {attempt}/{attempts} failed: {exc}",
                    flush=True,
                )
                if attempt == attempts:
                    raise
                time.sleep(15 * attempt)

    sources_used: list[str] = []
    lines: list[str] = []

    # ── Source A: WaxalNLP aka_asr train transcripts ──
    # NOTE (deviation from datasets streaming): streaming via
    # load_dataset(streaming=True) downloads the audio bytes for every row
    # (~2000 rows in 13 min observed — 20k would take hours), and the
    # datasets-server /rows API is currently broken for this dataset (500s).
    # Instead we read ONLY the text column from the repo's native parquet
    # shards via HfFileSystem HTTP range requests (pyarrow column projection)
    # — validated at ~2000 transcripts / 90s per shard. Same dataset, same
    # split, same 20000 cap, same column detection.
    def _collect_waxal_shard(shard: str, remaining: int) -> list[str]:
        import fsspec
        import pyarrow.parquet as pq

        url = (
            "https://huggingface.co/datasets/google/WaxalNLP/resolve/main/" + shard
        )
        fs = fsspec.filesystem("https")
        out: list[str] = []
        with fs.open(url, "rb") as f:
            pf = pq.ParquetFile(f)
            cols = pf.schema_arrow.names
            text_col = _find_text_col(cols)
            print(
                f"[twi-lm] shard {shard.split('/')[-1]} rows={pf.metadata.num_rows} "
                f"text={text_col}",
                flush=True,
            )
            if text_col is None:
                raise RuntimeError(f"No transcript column found: {cols}")
            for batch in pf.iter_batches(batch_size=1000, columns=[text_col]):
                for v in batch.column(0):
                    t = _normalize_text(str(v.as_py() or ""))
                    if len(t) >= 2:
                        out.append(t)
                if len(out) >= remaining:
                    break
        return out[:remaining]

    def _collect_waxal() -> list[str]:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        shards = sorted(
            p
            for p in api.list_repo_files("google/WaxalNLP", repo_type="dataset")
            if p.startswith("data/ASR/aka/aka-train-") and p.endswith(".parquet")
        )
        print(
            f"[twi-lm] reading Waxal aka train text column from {len(shards)} "
            f"parquet shards limit={waxal_limit}",
            flush=True,
        )
        out: list[str] = []
        for shard in shards:
            if waxal_limit and len(out) >= waxal_limit:
                break
            remaining = waxal_limit - len(out) if waxal_limit else 10**9
            texts = _with_retries(
                lambda s=shard, r=remaining: _collect_waxal_shard(s, r),
                f"waxal-shard:{shard.split('/')[-1]}",
            )
            out.extend(texts)
            print(f"[twi-lm] waxal transcripts={len(out)}", flush=True)
        print(f"[twi-lm] waxal total={len(out)}", flush=True)
        return out

    waxal_lines = _with_retries(_collect_waxal, "waxal-stream")
    if waxal_lines:
        lines.extend(waxal_lines)
        sources_used.append(f"google/WaxalNLP:aka_asr(n={len(waxal_lines)})")

    # ── Source B: local recorder manifest references ──
    for manifest_path in (
        "/root/gha_local_asr/manifest.jsonl",
        "/root/gha_local_asr/manifest.train32.jsonl",
    ):
        if not os.path.exists(manifest_path):
            print(f"[twi-lm] local manifest not found: {manifest_path}", flush=True)
            continue
        local_lines: list[str] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                t = _normalize_text(str(item.get("reference") or item.get("text") or ""))
                if len(t) >= 2:
                    local_lines.append(t)
        lines.extend(local_lines)
        sources_used.append(
            f"local:{os.path.basename(manifest_path)}(n={len(local_lines)})"
        )
        print(
            f"[twi-lm] local manifest {manifest_path} transcripts={len(local_lines)}",
            flush=True,
        )

    # Dedupe (keep order) so repeated utterances don't skew n-gram counts.
    seen: set[str] = set()
    corpus = [l for l in lines if not (l in seen or seen.add(l))]
    print(
        f"[twi-lm] corpus lines={len(corpus)} (raw={len(lines)}, deduped={len(lines) - len(corpus)})",
        flush=True,
    )
    if len(corpus) < 100:
        raise RuntimeError(f"corpus too small to train an LM: {len(corpus)} lines")

    vocab = {w for l in corpus for w in l.split()}
    n_tokens = sum(len(l.split()) for l in corpus)

    corpus_path = "/tmp/twi_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus) + "\n")

    # ── Train 3-gram KenLM ──
    lmplz = "/tmp/kenlm/build/bin/lmplz"
    build_binary = "/tmp/kenlm/build/bin/build_binary"
    arpa_path = "/lm/twi_3gram.arpa"
    bin_path = "/lm/twi_3gram.bin"

    print("[twi-lm] running lmplz -o 3", flush=True)
    with open(corpus_path, "r", encoding="utf-8") as src, open(
        arpa_path, "w", encoding="utf-8"
    ) as dst:
        subprocess.run([lmplz, "-o", "3"], stdin=src, stdout=dst, check=True)
    print(f"[twi-lm] wrote {arpa_path}", flush=True)

    print("[twi-lm] running build_binary", flush=True)
    subprocess.run([build_binary, arpa_path, bin_path], check=True)
    print(f"[twi-lm] wrote {bin_path}", flush=True)

    arpa_bytes = os.path.getsize(arpa_path)
    bin_bytes = os.path.getsize(bin_path)

    card = f"""# Twi 3-gram KenLM (ghana-health-ai)

KenLM language model for CTC beam-search decoding of the DONDO w2v-BERT Twi ASR
checkpoints. Load via `train_dondo_asr.py --lm-path <path to twi_3gram.arpa or .bin>`
(pyctcdecode `BeamSearchDecoderCTC`).

## Artifacts (Modal volume `akan-speech-lm`)

- `twi_3gram.arpa` — ARPA text model ({arpa_bytes} bytes)
- `twi_3gram.bin` — kenlm binary (probing) model ({bin_bytes} bytes)

## Corpus stats

- Lines (after dedupe): {len(corpus)}
- Raw lines collected: {len(lines)}
- Total tokens: {n_tokens}
- Vocab size (unique words): {len(vocab)}
- N-gram order: 3
- Normalization: same `_normalize_text` as `train_dondo_asr.py`
  (lowercase, NFC, punctuation stripped except Twi letters)

## Sources

{chr(10).join(f"- `{s}`" for s in sources_used)}
"""
    with open("/lm/LM_CARD.md", "w", encoding="utf-8") as f:
        f.write(card)
    lm_vol.commit()

    summary = {
        "status": "ok",
        "lines": len(corpus),
        "raw_lines": len(lines),
        "tokens": n_tokens,
        "vocab_size": len(vocab),
        "sources": sources_used,
        "arpa_bytes": arpa_bytes,
        "bin_bytes": bin_bytes,
    }
    print(summary, flush=True)
    print(card, flush=True)
    return summary


@app.local_entrypoint()
def main(waxal_limit: int = WAXAL_CAP):
    summary = build_twi_lm.remote(waxal_limit=waxal_limit)
    print(f"[twi-lm] done: {summary}")
