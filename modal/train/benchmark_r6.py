"""Thin capability benchmark for R6 — load base/R5/R6, run benchmark datasets, score.

Like testing an API: load the model, send each benchmark's questions, score the answers.
Twi coverage: AfriMMLU (MCQ) + AfriQA (generative QA) + SIB-200 (classification).
General: MMLU, HellaSwag, ARC-Challenge. Health proxy: MedQA, PubMedQA (English).
FLORES-200 (Twi MT) is gated on HF — needs an auth token, noted but not run here.
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers==5.17.0", "peft==0.20.0", "torch==2.13.0", "accelerate==1.15.0",
    "datasets>=3.0", "evaluate>=0.4", "sacrebleu", "sentencepiece",
).env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"})

app = modal.App("ghana-health-r6-benchmark")
train_vol = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)
cache_vol = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)

MODEL = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REVISION = "693f12e444a30971ca53889128d7bade9491ecb3"
R5_ADAPTER = "/artifacts/understanding-adaptation-r5/understanding_20261110T103734Z/responses-adapter"
R6_ADAPTER = "/artifacts/understanding-adaptation-r6/understanding_20261113T152209Z/responses-adapter"

SIB200_LABELS = ["politics", "sports", "health", "entertainment", "science/technology",
                 "travel", "geography"]


def load_model(variant):
    import torch
    from transformers import Qwen3_5ForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir="/cache/hf",
                                        padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, revision=REVISION, cache_dir="/cache/hf",
                                               torch_dtype=torch.bfloat16, device_map="auto")
    if variant == "r5":
        model = PeftModel.from_pretrained(model, R5_ADAPTER)
    elif variant == "r6":
        model = PeftModel.from_pretrained(model, R6_ADAPTER)
    model.eval()
    return model, tok


def generate(model, tok, prompt, max_new=64):
    import torch
    from transformers import GenerationConfig
    EOS, PAD = 248046, 248044
    if tok.chat_template:
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
        inputs = enc["input_ids"].to("cuda")
    else:
        ids = tok.encode(prompt)
        inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new,
                              eos_token_id=EOS, pad_token_id=PAD, use_cache=True,
                              return_dict_in_generate=True)
    try:
        config.enable_thinking = False
    except Exception:
        pass
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                generation_config=config)
    generated = output.sequences[0, inputs.shape[1]:].tolist()
    return tok.decode(generated, skip_special_tokens=True).strip()


def first_letter(text):
    # Drop any thinking/reasoning preamble, then look for a leading letter or a
    # "N." / "N)" marker — the answer letter is what follows the prompt's "Answer:".
    import re
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # Strip markdown emphasis markers that may wrap the answer letter.
    text = text.replace("**", "").replace("*", "")
    # Prefer a standalone letter token at the start of a line (e.g. "B." or "B)").
    m = re.search(r"(?m)^\s*([A-Fa-f])[\s.):]", text)
    if m:
        return m.group(1).upper()
    # Fall back: first A-F character after any leading prose.
    for ch in text:
        if ch.upper() in "ABCDEF":
            return ch.upper()
    return (text[:1] or "?").upper()


def mcq(model, tok, rows, limit):
    """rows: iterable of (prompt, correct_letter)."""
    correct = total = 0
    samples = []
    for i, (prompt, answer) in enumerate(rows):
        if limit and i >= limit:
            break
        pred = generate(model, tok, prompt)
        total += 1
        got = first_letter(pred) == answer.upper()
        correct += got
        if len(samples) < 3:
            samples.append({"pred": pred[:60], "got_letter": first_letter(pred),
                            "answer": answer, "match": got})
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else None, "samples": samples}


def _mcq_prompt(question, opts):
    return question + "\n\n" + "\n".join(f"{l}. {o}" for l, o in zip("ABCD", opts)) + "\nAnswer:"


def bench_afrimmlu(model, tok, limit):
    from datasets import load_dataset
    ds = load_dataset("masakhane/afrimmlu", "twi", split="test")
    def rows():
        for r in ds:
            yield _mcq_prompt(r["question"], r["choices"]), r["answer"]
    return mcq(model, tok, rows(), limit)


def bench_afriqa(model, tok, limit):
    """Generative short-answer QA; exact-match on a normalized answer."""
    from datasets import load_dataset
    import re
    ds = load_dataset("masakhane/afriqa", "twi", split="validation")
    correct = total = 0
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        pred = generate(model, tok, f"Answer briefly:\n\n{r['question']}")
        golds = [g.strip().lower() for g in r["answers"]]
        p = re.sub(r"[^a-z0-9 ]", "", pred.lower()).strip()
        total += 1
        correct += any(g in p or p in g for g in golds)
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else None}


def bench_mmlu(model, tok, limit):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    def rows():
        for r in ds:
            yield _mcq_prompt(r["question"], r["choices"]), "ABCD"[r["answer"]]
    return mcq(model, tok, rows(), limit)


def bench_hellaswag(model, tok, limit):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    def rows():
        for r in ds:
            yield _mcq_prompt(r["ctx"], r["endings"]), "ABCD"[int(r["label"])]
    return mcq(model, tok, rows(), limit)


def bench_arc(model, tok, limit):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    def rows():
        for r in ds:
            opts = r["choices"]["text"]
            yield _mcq_prompt(r["question"], opts), r["answerKey"]
    return mcq(model, tok, rows(), limit)


def bench_medqa(model, tok, limit):
    from datasets import load_dataset
    ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa", split="test")
    def rows():
        for r in ds:
            opts = r["choices"]
            ans_text = (r["answer"][0] if isinstance(r["answer"], list) else str(r["answer"])).strip()
            letter = next((l for l, o in zip("ABCD", opts) if o.strip() == ans_text), None)
            if letter is None:
                continue
            yield _mcq_prompt(r["question"], opts), letter
    return mcq(model, tok, rows(), limit)


def bench_pubmedqa(model, tok, limit):
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    def rows():
        for r in ds:
            ctx = " ".join(r["context"]["contexts"][:3])[:1500]
            letter = "ABC"[["yes", "no", "maybe"].index(r["final_decision"])]
            yield (f"Context: {ctx}\n\nQuestion: {r['question']}\n\n"
                   "A. yes\nB. no\nC. maybe\nAnswer:", letter)
    return mcq(model, tok, rows(), limit)


def bench_sib200(model, tok, limit):
    from datasets import load_dataset
    ds = load_dataset("davlan/sib200", "twi_Latn", split="test")
    correct = total = 0
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        pred = generate(model, tok, "Classify into one of: " + ", ".join(SIB200_LABELS)
                        + f"\n\nText: {r['text']}\n\nCategory:")
        total += 1
        label = SIB200_LABELS[r["category"]] if isinstance(r["category"], int) else str(r["category"])
        correct += label.lower() in pred.lower()
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else None}


BENCHMARKS = (("afrimmlu_twi", bench_afrimmlu),
              ("sib200_twi", bench_sib200), ("mmlu", bench_mmlu),
              ("hellaswag", bench_hellaswag), ("arc_challenge", bench_arc),
              ("medqa", bench_medqa), ("pubmedqa", bench_pubmedqa))


@app.function(gpu="H100", image=image, volumes={"/cache": cache_vol, "/artifacts": train_vol},
              timeout=7200)
def benchmark(variant: str = "r6", limit: int = 0):
    model, tok = load_model(variant)
    limit = limit or 0
    results = {"variant": variant}
    for name, fn in BENCHMARKS:
        try:
            results[name] = fn(model, tok, limit)
            print(f"[{variant}] {name}: {results[name]}", flush=True)
        except Exception as e:
            import traceback
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{variant}] {name}: ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    return results


@app.local_entrypoint()
def main(variant: str = "r6", limit: int = 0):
    print(benchmark.remote(variant, limit))
