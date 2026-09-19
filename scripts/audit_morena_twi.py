"""Audit MORENA tokenization on the frozen Ghana Health AI Twi source pool.

This is diagnostic only: no training, labels, uploads, or corpus mutation.
It reuses audit_twi_pretraining.py's source/protection logic and compares MORENA
against the already-pinned Qwen/Gemma tokenizers.

Usage:
  python scripts/audit_morena_twi.py --download
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

import audit_twi_pretraining as corpus

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/response-adaptation/twi-text-audit-sources.v1.json"
DEFAULT_OUT = ROOT / "tmp/morena-twi-audit/v1"
MORENA_REPO = "vamboai/morena-1.5b-base"


def words(text: str) -> int:
    return len(corpus.normalized(text).split())


def tokenizer_spec(name: str, repo: str, revision: str) -> dict:
    path = hf_hub_download(repo, "tokenizer.json", revision=revision, token=False)
    return {
        "name": name,
        "repo": repo,
        "revision": revision,
        "tokenizer": Tokenizer.from_file(path),
        "tokenizer_sha256": corpus.sha(path),
    }


def summarize(values):
    if not values:
        return {}
    ordered = sorted(values)
    def pct(p):
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": pct(.90),
        "p95": pct(.95),
        "max": max(values),
    }


def run(output: Path, download: bool, limit: int | None):
    if output.exists():
        raise ValueError("Choose a new output folder; audit outputs are immutable")

    registry = json.loads(REGISTRY.read_text())
    resolved = HfApi().model_info(MORENA_REPO).sha
    specs = [tokenizer_spec("morena15b", MORENA_REPO, resolved)]
    for item in registry["tokenizers"]:
        specs.append(tokenizer_spec(item["name"], item["repo"], item["revision"]))

    protected = corpus.protected_texts()
    blocked = corpus.OverlapIndex(protected) if hasattr(corpus, "OverlapIndex") else None
    # OverlapIndex is defined in build_language_adaptation, as in the parent audit.
    if blocked is None:
        from build_language_adaptation import OverlapIndex
        blocked = OverlapIndex(protected)

    rows = []
    for raw in corpus.sources(registry, download):
        text = raw["text"]
        if corpus.text_flags(text, registry["limits"]):
            continue
        if raw["upstream_split"] in ("test", "validation") or blocked.contains(text):
            continue
        rows.append(raw)
        if limit and len(rows) >= limit:
            break

    output.mkdir(parents=True)
    metrics = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(lambda: defaultdict(float))
    examples = []

    batch_size = 256
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [unicodedata.normalize("NFC", r["text"]) for r in batch]
        for spec in specs:
            enc = spec["tokenizer"].encode_batch(texts, add_special_tokens=False)
            for row, text, encoded in zip(batch, texts, enc):
                n_tokens = len(encoded.ids)
                n_words = max(1, words(text))
                n_bytes = max(1, len(text.encode("utf-8")))
                key = row["source"]
                metrics[key][spec["name"] + ":tokens_per_word"].append(n_tokens / n_words)
                metrics[key][spec["name"] + ":tokens_per_byte"].append(n_tokens / n_bytes)
                metrics[key][spec["name"] + ":bytes_per_token"].append(n_bytes / max(1, n_tokens))
                totals[key][spec["name"] + ":tokens"] += n_tokens
                totals[key][spec["name"] + ":bytes"] += n_bytes
                totals[key][spec["name"] + ":words"] += n_words
                totals[key][spec["name"] + ":rows"] += 1

        if len(examples) < 100:
            for row, text in zip(batch, texts):
                item = {"id": row["id"], "source": row["source"], "text": text}
                for spec in specs:
                    e = spec["tokenizer"].encode(text, add_special_tokens=False)
                    item[spec["name"]] = {"count": len(e.ids), "tokens": e.tokens[:80]}
                examples.append(item)
                if len(examples) >= 100:
                    break

    report = {
        "stage": "tokenizer_audit_only_no_training",
        "model": MORENA_REPO,
        "resolved_morena_revision": resolved,
        "rows": len(rows),
        "protected_text_count": len(protected),
        "tokenizers": [{k: v for k, v in s.items() if k != "tokenizer"} for s in specs],
        "by_source": {},
        "limitations": [
            "Tokenizer efficiency is not evidence of semantic understanding.",
            "This run does not evaluate model loss, generation, translation, QA, or health safety.",
            "Only structurally eligible, non-protected text is measured; no private chats or audio are imported.",
            "Source row counts are not independent dialogue counts; dictionary and synthetic mixtures remain distinct.",
        ],
    }
    for source in sorted(metrics):
        report["by_source"][source] = {
            "distribution": {name: summarize(vals) for name, vals in sorted(metrics[source].items())},
            "totals": dict(sorted(totals[source].items())),
        }

    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (output / "tokenization-examples.jsonl").open("w") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "morena_revision": resolved}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.output, args.download, args.limit)
