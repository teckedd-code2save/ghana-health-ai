"""Pinned R2 English validation and native R5 retention loss, never training data."""
import hashlib
import json
import math
from pathlib import Path

R2_PACK_SHA = "1e632f4d7382d6b7db910973a83d2483d4e1f843659335951fe59f81bd974084"
ENGLISH_SHA = "40a6012a7055efdbb79767e4ea27f8abb560057217f964968291caa39dc6f038"
ENGLISH = "english_oasst_final_assistant"
TWI = "twi_native_causal"
SOURCE_FILES = ("r2-manifest.json", "snapshot-manifest.json", "batches.json", "validation.jsonl")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sources(pack, validation):
    require(pack is not None and validation is not None,
            "Language loss requires --r2-pack and --english-validation (all 44 heldout OASST rows)")
    pack = Path(pack)
    return dict(zip(SOURCE_FILES, (pack / "manifest.json", pack / "protection/snapshot/manifest.json",
                                  pack / "protection/snapshot/batches.json", Path(validation)), strict=True))


def load(paths):
    """Minimal source chain: pinned pack -> snapshot -> batch -> exact validation file."""
    require(set(paths) == set(SOURCE_FILES), "Retention source inventory differs")
    require(all(not Path(p).is_symlink() for p in paths.values()), "Linked retention source")
    require(sha(paths["r2-manifest.json"]) == R2_PACK_SHA, "R2 retention manifest differs")
    read = lambda name: json.loads(Path(paths[name]).read_text())
    pack = read("r2-manifest.json")
    snapshot_sha = sha(paths["snapshot-manifest.json"])
    require(snapshot_sha == pack["snapshot_manifest_sha256"]
            == pack["artifacts"]["protection/snapshot/manifest.json"], "R2 retention snapshot differs")
    snapshot = read("snapshot-manifest.json")
    require(sha(paths["batches.json"]) == snapshot["artifacts"]["batches.json"]
            == pack["artifacts"]["protection/snapshot/batches.json"], "R2 retention batches differ")
    batches = [b for b in read("batches.json") if b["batch_id"] == "english-v3-20260929"]
    require(len(batches) == 1 and batches[0]["files"]["english-retention/validation.jsonl"] == ENGLISH_SHA
            == sha(paths["validation.jsonl"]), "English validation source differs")
    rows = [json.loads(line) for line in Path(paths["validation.jsonl"]).read_text().splitlines()]
    require(len(rows) == len({r["id"] for r in rows}) == 44
            and all(r["split"] == "validation" and r["language"] == "en"
                    and r["record_type"] == "conversation" and r["origin"] == "source_preserved" for r in rows),
            "English retention must contain the original 44 validation conversations")
    held = [{"id": r["id"], "loss_cohort": ENGLISH, "messages": r["messages"],
             "source_row_sha256": digest(r)} for r in rows]
    proof = {"version": "r5-retention-source-v1", "cohort": ENGLISH, "rows": 44,
             "split": "validation", "selected_test_rows": 0, "training_eligible": False,
             "source_files": {name: sha(path) for name, path in paths.items()},
             "batch_id": batches[0]["batch_id"], "row_inventory_sha256": digest([
                 {"id": r["id"], "source_row_sha256": r["source_row_sha256"]} for r in held]),
             "objective": "native_chat_final_assistant_mask_including_eos",
             "truncate": False, "pool_with_twi_causal_loss": False}
    return held, proof


def cohort(row):
    value = row.get("loss_cohort", TWI)
    require(value in {TWI, ENGLISH}, "Unknown retention cohort")
    return value


def request(row):
    return {"row_id": row["id"], "cohort": cohort(row), "source_row_sha256": digest(row)}


def encode(tokenizer, row, eos):
    if cohort(row) == ENGLISH:
        from check_afrique_training_contract import encode_complete
        encoded = encode_complete(tokenizer, row, 4096)
    else:
        ids = tokenizer.encode(row["text"], add_special_tokens=False) + [eos]
        encoded = {"input_ids": ids, "labels": ids, "attention_mask": [1] * len(ids)}
    ids, labels = encoded["input_ids"], encoded["labels"]
    require(1 < len(ids) <= 4096 and len(ids) == len(labels) == len(encoded["attention_mask"])
            and ids[-1] == labels[-1] == eos and all(m == 1 for m in encoded["attention_mask"])
            and all(label == -100 or label == token for token, label in zip(ids, labels, strict=True)),
            "Invalid native retention encoding; no truncation")
    predicted = sum(label != -100 for label in labels[1:])
    require(predicted > 0, "No shifted retention targets")
    return encoded, {"token_ids_sha256": digest(ids), "labels_sha256": digest(labels),
                     "sequence_tokens": len(ids), "predicted_tokens": predicted}


def shifted_masked_nll(logits, labels):
    from torch.nn import functional as F
    require(logits.shape[:2] == labels.shape and labels.shape[0] == 1 and labels.shape[1] > 1,
            "Invalid masked-loss shapes")
    count = int((labels[0, 1:] != -100).sum().item())
    require(count > 0, "No shifted assistant targets")
    total = 0.0
    # Chunk the vocabulary projection loss, as in the pinned causal-loss runtime.
    for first in range(0, labels.shape[1] - 1, 128):
        end = min(first + 128, labels.shape[1] - 1)
        total += F.cross_entropy(logits[0, first:end].float(), labels[0, first+1:end+1],
                                 ignore_index=-100, reduction="sum").item()
    require(math.isfinite(total) and total >= 0, "Nonfinite assistant loss")
    return {"predicted_tokens": count, "summed_nll": total}


def measure(runtime, row, variant, encoded):
    if cohort(row) == TWI:
        return runtime.language_loss(row["text"], variant)
    import torch
    with runtime.select(variant), torch.inference_mode():
        inputs = torch.tensor([encoded["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([encoded["labels"]], dtype=torch.long, device="cuda")
        logits = runtime.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
        return shifted_masked_nll(logits, labels)
