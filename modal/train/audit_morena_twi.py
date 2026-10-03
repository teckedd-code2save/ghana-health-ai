"""Modal Gate 0 for MORENA as a Twi substrate.

Runs a bounded tokenizer audit directly against pinned public Twi sources.
No GPU, training, labels, private conversations, or model upload.

  modal run modal/train/audit_morena_twi.py
  modal run modal/train/audit_morena_twi.py --limit 5000
"""
from __future__ import annotations

import json
from pathlib import Path
import modal

app = modal.App("ghana-morena-twi-audit")
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==0.35.3", "tokenizers==0.22.1", "pyarrow==21.0.0")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1"})
)

# Pinned public sources already audited by the project.
SOURCES = [
    {
        "name": "pristine_twi",
        "repo": "ghananlpcommunity/pristine-twi",
        "revision": "9aca5772a8bec63aafb3008dc2fc0df86227cb19",
        "files": [
            "data/train-00000-of-00004.parquet",
            "data/train-00001-of-00004.parquet",
            "data/train-00002-of-00004.parquet",
            "data/train-00003-of-00004.parquet",
        ],
        "text": "twi",
        "origin": "synthetic_gemini_news_conditioned",
    },
]
BASELINES = [
    ("qwen35", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("gemma4", "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
]
MORENA = "vamboai/morena-1.5b-base"


@app.function(image=image, cpu=8, memory=16384, timeout=3600, retries=0,
              max_containers=1, volumes={"/cache": cache, "/artifacts": artifacts})
def audit(limit: int = 0):
    import hashlib
    import statistics
    import time
    import unicodedata
    from collections import defaultdict
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer

    resolved = HfApi().model_info(MORENA).sha
    specs = [("morena15b", MORENA, resolved), *BASELINES]
    toks = {}
    evidence = []
    for name, repo, revision in specs:
        path = hf_hub_download(repo, "tokenizer.json", revision=revision, token=False)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        toks[name] = Tokenizer.from_file(path)
        evidence.append({"name": name, "repo": repo, "revision": revision, "tokenizer_sha256": digest})

    stats = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(lambda: defaultdict(int))
    samples = []
    count = 0

    def normalized_words(text):
        return max(1, len(" ".join(text.casefold().split()).split()))

    for source in SOURCES:
        for filename in source["files"]:
            local = hf_hub_download(source["repo"], filename, repo_type="dataset",
                                    revision=source["revision"], token=False)
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=256, columns=[source["text"]]):
                texts = [unicodedata.normalize("NFC", str(v.as_py() or "")).strip() for v in batch.column(0)]
                texts = [t for t in texts if t and len(t.split()) >= 5]
                if limit:
                    texts = texts[:max(0, limit - count)]
                if not texts:
                    break
                encoded = {name: tok.encode_batch(texts, add_special_tokens=False) for name, tok in toks.items()}
                for i, text in enumerate(texts):
                    b = max(1, len(text.encode("utf-8"))); w = normalized_words(text)
                    for name in toks:
                        n = len(encoded[name][i].ids)
                        stats[source["name"]][name + ":tokens_per_word"].append(n / w)
                        stats[source["name"]][name + ":tokens_per_byte"].append(n / b)
                        stats[source["name"]][name + ":bytes_per_token"].append(b / max(1, n))
                        totals[source["name"]][name + ":tokens"] += n
                    if len(samples) < 100:
                        samples.append({"source": source["name"], "text": text,
                            **{name: {"count": len(encoded[name][i].ids),
                                      "tokens": encoded[name][i].tokens[:80]} for name in toks}})
                count += len(texts)
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break
        if limit and count >= limit:
            break

    def summary(values):
        values = sorted(values)
        def pct(p): return values[min(len(values)-1, int((len(values)-1)*p))]
        return {"mean": statistics.fmean(values), "median": statistics.median(values),
                "p90": pct(.90), "p95": pct(.95), "max": max(values)}

    run_id = "morena_twi_tokenizer_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = Path("/artifacts/morena-twi-audit") / run_id
    dest.mkdir(parents=True, exist_ok=False)
    report = {
        "stage": "gate0_public_tokenizer_audit_only",
        "run_id": run_id,
        "rows": count,
        "morena_model": MORENA,
        "resolved_morena_revision": resolved,
        "tokenizers": evidence,
        "sources": SOURCES,
        "by_source": {
            source: {"distribution": {k: summary(v) for k, v in values.items()},
                     "totals": dict(totals[source])}
            for source, values in stats.items()
        },
        "limitations": [
            "Tokenizer efficiency is not semantic understanding.",
            "Gate 0 uses pinned public Pristine Twi only; the frozen local stage corpus remains the broader decision set.",
            "Pristine is synthetic Gemini news-conditioned Twi and must not stand in for conversational, speech, or health Twi.",
            "No training, generation, private data, protected evaluation, upload, or production promotion occurred.",
        ],
    }
    (dest / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (dest / "samples.jsonl").open("w") as f:
        for row in samples: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    artifacts.commit(); cache.commit()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main(limit: int = 0):
    print(json.dumps(audit.remote(limit=limit), ensure_ascii=False, indent=2))
