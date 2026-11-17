"""CPU-only benchmark contracts; source parsing never depends on model output."""
import ast
import hashlib
import json
import math
import re
import string
import unicodedata

VERSION = "r6-benchmark-v2"
VARIANTS = ("base", "r5", "r6")
SIB_LABELS = ("politics", "sports", "health", "entertainment", "science/technology", "travel", "geography")
SOURCES = {
    "afrimmlu_twi": ("masakhane/afrimmlu", "twi", "test"),
    "sib200_twi": ("davlan/sib200", "twi_Latn", "test"),
    "mmlu": ("cais/mmlu", "all", "test"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "medqa": ("bigbio/med_qa", "med_qa_en_4options_bigbio_qa", "test"),
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_labeled", "train"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def choices(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    require(isinstance(value, (list, tuple)) and 2 <= len(value) <= 26, "Invalid option list")
    require(all(isinstance(v, str) and v.strip() for v in value), "Empty/non-text option")
    return list(value)


def mcq_row(question, options, answer, labels=None):
    options = choices(options)
    letters = list(string.ascii_uppercase[:len(options)])
    labels = letters if labels is None else [str(v) for v in labels]
    require(len(labels) == len(options) and len(set(labels)) == len(labels), "Invalid source labels")
    require(str(answer) in labels, "Answer absent from presented choices")
    require(isinstance(question, str) and question.strip(), "Empty question")
    return {"task": "mcq", "prompt": question + "\n\n" + "\n".join(
        f"{letter}. {option}" for letter, option in zip(letters, options)) + "\nAnswer:",
        "options": options, "labels": letters, "gold": letters[labels.index(str(answer))]}


def normalize(name, row, index):
    if name == "afrimmlu_twi":
        result = mcq_row(row["question"], row["choices"], row["answer"])
    elif name == "mmlu":
        result = mcq_row(row["question"], row["choices"], row["answer"], range(len(row["choices"])))
    elif name == "hellaswag":
        result = mcq_row(row["ctx"], row["endings"], row["label"], range(len(row["endings"])))
    elif name == "arc_challenge":
        result = mcq_row(row["question"], row["choices"]["text"], row["answerKey"], row["choices"]["label"])
    elif name == "medqa":
        opts = choices(row["choices"])
        answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        matches = [i for i, option in enumerate(opts) if option.strip() in {v.strip() for v in answers}]
        require(len(matches) == 1, "MedQA answer does not map uniquely; do not silently skip")
        result = mcq_row(row["question"], opts, matches[0], range(len(opts)))
    elif name == "pubmedqa":
        # Retain the historical context budget; recorded in the manifest.
        context = " ".join(row["context"]["contexts"][:3])[:1500]
        result = mcq_row(f"Context: {context}\n\nQuestion: {row['question']}",
                         ["yes", "no", "maybe"], row["final_decision"], ["yes", "no", "maybe"])
    elif name == "sib200_twi":
        gold = row["category"]
        require(isinstance(gold, str) and gold in SIB_LABELS, "Unknown SIB label encoding")
        result = {"task": "classification_generation", "prompt": "Classify into one of: "
                  + ", ".join(SIB_LABELS) + f"\n\nText: {row['text']}\n\nCategory:", "gold": gold}
    else:
        raise ValueError("Unknown benchmark")
    result.update({"id": f"{name}:{index}", "source_index": index, "source_sha256": digest(row),
                   "subject": row.get("subject", row.get("activity_label", "unspecified"))})
    result["input_sha256"] = digest(result)
    return result


def classify_output(text):
    """Only one explicit category is valid, never a gold-label substring match."""
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = re.sub(r"^(?:category|label)\s*:\s*", "", text)
    text = text.strip(" \n\t.*`\"'")
    return text if text in SIB_LABELS else None


def choose(scores, labels):
    require(set(scores) == set(labels) and len(labels) == len(set(labels)), "Wrong scored choices")
    require(all(math.isfinite(v) for v in scores.values()), "Nonfinite option score")
    return max(labels, key=lambda label: scores[label])


def answer_tokens(tokenizer, prefix, labels):
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    continuations = {}
    for label in labels:
        combined = tokenizer.encode(prefix + label, add_special_tokens=False)
        require(combined[:len(prompt_ids)] == prompt_ids, "Answer boundary changes prompt tokens")
        continuation = combined[len(prompt_ids):]
        require(continuation, "Empty answer encoding")
        continuations[label] = continuation
    require(len({tuple(ids) for ids in continuations.values()}) == len(labels), "Nonunique answer encodings")
    return prompt_ids, continuations


def summarize(rows):
    require(rows and len({r['id'] for r in rows}) == len(rows), "Empty/duplicate evidence")
    total = len(rows)
    return {"correct": sum(r["correct"] for r in rows), "total": total,
            "accuracy": sum(r["correct"] for r in rows) / total,
            "invalid": sum(r["prediction"] is None for r in rows),
            "chance": sum(1 / len(r.get("labels", SIB_LABELS)) for r in rows) / total}
