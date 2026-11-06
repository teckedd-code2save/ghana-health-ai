"""Shared validation, token-weighted updates and paired scoring for Stage 1."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_pack(folder, require_live=True):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    digest = sha(folder / "manifest.json")
    if json.loads((folder / "PREPARED.json").read_text())["manifest_sha256"] != digest:
        raise ValueError("Preparation seal changed")
    for name, checksum in manifest["artifacts"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or sha(path) != checksum:
            raise ValueError("Input changed: " + name)
    if require_live:
        if manifest["review_mode"] != "live":
            raise ValueError("Saved reviews cannot launch training")
        review = json.loads((folder / "review-snapshot.json").read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(review["fetched_at"])).total_seconds()
        if not 0 <= age <= 86400:
            raise ValueError("Refresh reviews before launch (24-hour maximum)")
    from corpus_record_types import POLICY, validate_example_type
    if manifest.get("record_type_policy") != POLICY:
        raise ValueError("Missing/stale record-type assessment; old preparations cannot launch")
    recipe = json.loads((folder / "experiment.json").read_text())
    if (recipe["experiment"] != "qwen9_source_alignment_v1" or recipe["max_steps"] != 200 or
            recipe["effective_batch"] != 16 or recipe["max_gpu_seconds"] != 5400 or recipe["production_eligible"]):
        raise ValueError("Unexpected bounded experiment")
    encoded, sources, groups = {}, {}, {}
    for split in ("train", "validation"):
        sources[split] = rows(folder / (split + ".jsonl"))
        encoded[split] = rows(folder / (split + ".tokens.jsonl"))
        if len({r["id"] for r in sources[split]}) != len(sources[split]):
            raise ValueError("Duplicate source view")
        for source, token in zip(sources[split], encoded[split], strict=True):
            if source.get("bucket", "").startswith("sentences_"):
                validate_example_type(source)
            if source["id"] != token["id"] or source["split"] != split:
                raise ValueError("Token order or split changed")
            n, labels, ids = source["supervised_tokens"], token["labels"], token["input_ids"]
            if not 0 < n < len(ids) or labels != [-100] * (len(ids) - n) + ids[-n:]:
                raise ValueError("Incorrect final-assistant loss mask")
            if groups.setdefault(source["group_id"], split) != split:
                raise ValueError("Source group crosses split")
    if len(encoded["train"]) != 3200 or len(encoded["validation"]) != 498:
        raise ValueError("Replan if reviewed corpus coverage changes; never silently shrink the run")
    return manifest, recipe, sources, encoded


def target_tokens(row):
    return sum(token != -100 for token in row["labels"][1:])


def backward_window(model, window, collator, device):
    """Normalize over all answer tokens in the effective batch, not each row."""
    import torch
    total = sum(target_tokens(row) for row in window)
    if total <= 0:
        raise ValueError("Empty optimizer window")
    observed = 0.0
    for row in window:
        batch = {k: value.to(device) for k, value in collator([row]).items()}
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        weighted = loss * (target_tokens(row) / total)
        weighted.backward()
        observed += float(weighted.detach())
    return observed, total


def flags(text):
    words = re.findall(r"\w+", text.casefold())
    repetition = max(Counter(tuple(words[i:i + 8]) for i in range(len(words) - 7)).values(), default=0)
    return {"empty": not text.strip(), "repetition": repetition >= 4, "max_repeated_eight_word_span": repetition}


def stop_details(ids, eos):
    if eos in ids:
        ids = ids[:ids.index(eos) + 1]
        return ids, "eos"
    return ids, "length"


def paired_metrics(base, adapter, losses, expected_alignment_ids, expected_development_ids):
    from sacrebleu.metrics import CHRF
    variants = {}
    expected = set(expected_alignment_ids) | set(expected_development_ids)
    for name, data in (("base", base), ("adapter", adapter)):
        indexed = {r["id"]: r for r in data}
        if len(indexed) != len(data) or set(indexed) != expected:
            raise ValueError("Incomplete or duplicate paired evaluation")
        variants[name] = indexed
    for key in expected:
        a, b = variants["base"][key], variants["adapter"][key]
        for field in ("messages", "tools", "reference", "bucket"):
            if a.get(field) != b.get(field):
                raise ValueError("Paired comparison inputs differ: " + field)
    chrf = CHRF(word_order=2)
    scores, mechanical = {}, {}
    for name, indexed in variants.items():
        scores[name] = {}
        for direction in ("sentences_to_tw", "sentences_to_en"):
            subset = [indexed[key] for key in sorted(expected_alignment_ids) if indexed[key]["bucket"] == direction]
            if len(subset) != 227:
                raise ValueError("Missing sentence direction")
            scores[name][direction] = chrf.corpus_score([r["prediction"] for r in subset], [[r["reference"] for r in subset]]).score
        subset = [indexed[key] for key in expected_development_ids]
        mechanical[name] = {flag: sorted(r["id"] for r in subset if flags(r["prediction"])[flag]) for flag in ("empty", "repetition")}
        mechanical[name]["length"] = sorted(r["id"] for r in subset if r["stop_reason"] == "length")
        mechanical[name]["inference_timeouts"] = sorted(key for key, r in indexed.items() if r["stop_reason"] == "time_limit")
    gains = {k: scores["adapter"][k] - scores["base"][k] for k in scores["base"]}
    english_ratio = losses["adapter"]["english_retention"]["loss"] / losses["base"]["english_retention"]["loss"]
    numerical = all(v >= 2 for v in gains.values()) and english_ratio <= 1.10
    numerical &= all(not (set(mechanical["adapter"][flag]) - set(mechanical["base"][flag])) for flag in ("empty", "repetition"))
    numerical &= not any(mechanical[v]["inference_timeouts"] for v in mechanical)
    return {"chrf_plus_plus": scores, "chrf_gains": gains, "english_loss_ratio": english_ratio,
        "development_flags": mechanical, "numerical_gate_passed": bool(numerical),
        "metric_signature": str(chrf.get_signature()), "native_review_required": True,
        "semantics_certified": False, "production_eligible": False}
