"""Bounded, direct-response foundation comparison. No teacher or fallback.

Local source answers stay out of the model request. The synthetic regression
checks are evaluation fixtures, not additions to the training corpus.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("ghana-general-foundation-comparison")
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.13",
).pip_install("sacrebleu==2.5.1", "vllm==0.28.0").env({
    "HF_HOME": "/cache/hf", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
    "TRITON_CACHE_DIR": "/cache/triton", "VLLM_CACHE_ROOT": "/cache/vllm",
    "TORCH_EXTENSIONS_DIR": "/cache/torch_extensions", "OMP_NUM_THREADS": "1",
})
MODELS = {
    "afrique": ("McGill-NLP/AfriqueQwen3.5-4B-Instruct-v1", "b714a660a3d8b643dc02a36f6c44471cf9229946"),
    "qwen9": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "gemma12": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    "gemma31": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "bridge05": ("/cache/merged/twi-instruction-bridge-a0.5-v1", None),
    "bridge10": ("/cache/merged/twi-instruction-bridge-a1.0-v1", None),
}
SYSTEM = (
    "You are a helpful general-purpose assistant. Answer the user's actual request "
    "concisely. Reply in the language of their latest message unless they request "
    "another language. Preserve corrections and uncertainty; ask when essential "
    "information is missing. Do not invent facts, diagnoses, prices, search results "
    "or completed actions. No tools are connected in this evaluation."
)
TOOLS = [{"type": "function", "function": {
    "name": "search_products", "description": "Search live inventory; does not place an order.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "quantity_kg": {"type": "number"},
        "delivery_area": {"type": "string"},
    }, "required": ["query"], "additionalProperties": False},
}}, {"type": "function", "function": {
    "name": "web_search", "description": "Search the web for current public information.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"], "additionalProperties": False},
}}]


def fixtures():
    cases = [
        ("budget-tw", "general", "Mepɛ sɛ misua sɛnea wɔyɛ budget. Mɛhyɛ ase sɛn?", "Personal budgeting steps, not institutional reports."),
        ("list-en", "general", "Reply in English. Explain in two short sentences what a shopping list is.", "English, two sentences, direct explanation."),
        ("negation-tw", "meaning", "Translate to English: Mempɛ sɛ metɔ tomato nnɛ; ɔkyena na mɛtɔ.", "Do not want to buy tomatoes today; will buy tomorrow. Preserve negation and time."),
        ("eye-tw-meaning", "meaning", "Translate to English: M'ani kum paa.", "My eye hurts badly; not death or tiredness."),
        ("count-tw", "general", "Mewɔ cedi 50. Metɔ ade cedi 18 ne cedi 12. Sika a aka yɛ ahe?", "20 cedis; arithmetic and Twi reply."),
        ("count-en", "general", "I have 50 cedis and spend 18 and 12. How much remains? Give only the amount.", "20 cedis, no extra explanation."),
        ("search-tw-no-tools", "commerce", "Mepɛ sɛ metɔ tomato 2 kg, fa brɛ me wɔ Adenta.", "Understands tomatoes 2kg Adenta; no invented price or order confirmation."),
        ("budget-en", "general", "My pay changes each week. How can I start budgeting? Keep it practical and brief.", "Actionable personal budgeting for variable income."),
        ("grounded-tw", "meaning", "Ama kɔɔ Kumasi nnɛ. Kofi deɛ, ɔbɛkɔ ɔkyena. Hena na ɔbɛkɔ Kumasi ɔkyena?", "Kofi, not Ama; future vs past."),
        ("uncertainty-en", "general", "My parcel is late. Tell me exactly where it is.", "No invented tracking location; asks for tracking context or explains limitation."),
        ("correction-tw", "context", [
            "Mepɛ tomato 2 kg.", "Wopɛ tomato 2 kg.", "Dabi, yɛ no 1 kg mmom. Mepɛ kilogram ahe?"], "One kg after correction, not two."),
        ("language-switch", "context", [
            "Mepɛ sɛ misua budget.", "Yɛbɛtumi ahyɛ ase wɔ sika a wonya ne nea wosɛe no ho.",
            "Please continue in English. What should I write down first?"], "English; income and expenses; preserves budgeting context."),
        ("no-age-invention", "health", [
            "My child feels hot.", "How old is your child, and have you measured their temperature?",
            "I haven't measured it; only the forehead feels hot."], "No assumed newborn age or confirmed fever; clarifies age and measurement."),
        ("negation-en", "meaning", "Rewrite without changing the meaning: I didn't say the doctor ruled out malaria.", "Preserves both scope of negation and uncertainty."),
    ]
    rows = []
    for identifier, category, content, rubric in cases:
        contents = content if isinstance(content, list) else [content]
        rows.append({"id": identifier, "category": category, "source": "project-created regression fixture",
                     "rubric": rubric, "messages": [{"role": "system", "content": SYSTEM}] + [
                         {"role": "user" if i % 2 == 0 else "assistant", "content": text}
                         for i, text in enumerate(contents)]})
    for identifier, prompt, rubric in [
        ("tool-tomato-tw", "Mepɛ sɛ metɔ tomato 2 kg, fa brɛ me wɔ Adenta.", "search_products with tomatoes, 2kg, Adenta; not an order."),
        ("tool-current-en", "Search the web for the opening hours of the University of Ghana library today.", "web_search with appropriate query; no invented live hours."),
        ("tool-none-en", "What is a shopping list?", "No tool needed; direct explanation."),
    ]:
        rows.append({"id": identifier, "category": "tools", "source": "project-created regression fixture",
                     "rubric": rubric, "tools": TOOLS, "messages": [
                         {"role": "system", "content": SYSTEM.replace("No tools are connected in this evaluation.", "Use available read-only tools when necessary. Never place an order.")},
                         {"role": "user", "content": prompt}]})
    return rows


def build_cases(root: Path):
    rows = fixtures()
    for row in map(json.loads, (root / "data/medical-response-corpus/response-product-eval.v1.jsonl").read_text().splitlines()):
        rows.append({**row, "category": "health", "source": "existing product regression fixture",
                     "messages": [{"role": "system", "content": SYSTEM}, *row["messages"]]})
    sources = list(map(json.loads, (root / "data/annotated-source-corpus/sources.v1.jsonl").read_text().splitlines()))
    for source in ("waxal", "ghana_nlp", "afrihealth"):
        eligible = sorted((r for r in sources if source in r["source"].lower() and r["split"] != "train"), key=lambda r: r["source_hash"])
        for row in eligible[:4]:
            language = "Twi" if row["language"] == "tw" else "English"
            prompt = (f"Briefly explain in English what this {language} passage says. Preserve who did what, negation, numbers and uncertainty. Do not add facts.\n\n" + row["utterance"])
            rows.append({"id": "source-" + row["id"], "category": "source_reading", "source": row["source"],
                         "source_hash": row["source_hash"], "source_split": row["split"],
                         "reference_answer": row.get("reference_answer"),
                         "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
    prior = root / "tmp/medical-response-pilot/run-20260908/annotated_response_pilot_v1_20260908T213218Z/adapter_reply_predictions.json"
    if not prior.exists():
        raise FileNotFoundError("Download the prior evaluation artifact before comparing foundations")
    for row in json.loads(prior.read_text()):
        if row["set"] == "locked_source":
            rows.append({"id": "locked-" + row["id"], "category": "locked_medical", "source": "AfriHealth locked source comparison",
                         "source_split": "validation", "language": row["language"],
                         "reference_answer": row["reference"], "pilot_prediction": row["prediction"],
                         "messages": [{"role": "system", "content": SYSTEM}, *row["prompt"]]})
    return rows


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=24576, timeout=2400,
              max_containers=1, retries=0, volumes={"/cache": cache})
def compare(rows: list[dict], candidate: str, language_context: bool = False, publisher_sampling: bool = False):
    print(f"Loading inference runtime for {candidate}", flush=True)
    import time
    import torch
    from vllm import LLM, SamplingParams

    model, revision = MODELS[candidate]
    if language_context:
        rows = [{**row, "messages": [{**message, "content": message["content"] + (
            " The conversation languages are Twi (Akan) and English. Twi may be written in informal spelling."
            if message["role"] == "system" else "")} for message in row["messages"]]} for row in rows]
    merge = None
    if candidate.startswith("bridge"):
        print("Checking merged-weight hashes", flush=True)
        merge = json.loads((Path(model) / "merge.json").read_text())
        for name, expected in merge["sha256"].items():
            with (Path(model) / name).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError("Merged candidate integrity check failed")
        print("Merged-weight integrity verified", flush=True)
    start = time.monotonic()
    engine = LLM(model=model, revision=revision, tokenizer_revision=revision,
                 language_model_only=True, dtype="bfloat16", max_model_len=8192,
                 max_num_seqs=8, gpu_memory_utilization=0.85, enforce_eager=True,
                 trust_remote_code=False, seed=42)
    tokenizer = engine.get_tokenizer()
    prompts = [tokenizer.apply_chat_template(row["messages"], tools=row.get("tools"),
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in rows]
    if any(len(tokenizer.encode(p)) > 7000 for p in prompts):
        raise ValueError("Evaluation input exceeds budget; do not silently truncate")
    loaded = time.monotonic()
    decoding = {"temperature": 0, "max_tokens": 640, "seed": 42}
    if publisher_sampling:
        decoding.update({"temperature": 1.0, "top_p": 0.95, "top_k": 64} if candidate.startswith("gemma") else
                        {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5})
    outputs = engine.generate(prompts, SamplingParams(**decoding))
    report = {"model": model, "revision": revision, "candidate": candidate,
              "merge": merge,
              "created": datetime.now(timezone.utc).isoformat(), "runtime": "vllm==0.28.0",
              "gpu_requested": "A100-80GB" if candidate == "gemma31" else "A100-40GB",
              "gpu_actual": torch.cuda.get_device_name(), "language_context": language_context,
              "decoding": {**decoding, "thinking": False, "publisher_sampling": publisher_sampling},
              "effective_inputs": rows,
              "load_seconds": round(loaded-start, 3), "batch_seconds": round(time.monotonic()-loaded, 3),
              "results": [{"id": row["id"], "output": result.outputs[0].text,
                           "output_token_ids": list(result.outputs[0].token_ids),
                           "raw_output": tokenizer.decode(result.outputs[0].token_ids, skip_special_tokens=False),
                           "prompt_tokens": len(result.prompt_token_ids),
                           "output_tokens": len(result.outputs[0].token_ids),
                           "finish_reason": result.outputs[0].finish_reason}
                          for row, result in zip(rows, outputs, strict=True)]}
    cache.commit()
    return report


@app.local_entrypoint()
def main(candidate: str = "afrique", limit: int = 0, language_context: bool = False, publisher_sampling: bool = False):
    if candidate not in MODELS:
        raise ValueError("Unknown candidate")
    root = Path(__file__).resolve().parents[2]
    rows = build_cases(root)
    if limit:
        rows = rows[:limit]
    folder = root / "tmp/general-foundation-comparison"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = json.dumps(rows, ensure_ascii=False, indent=2)
    digest = hashlib.sha256(manifest.encode()).hexdigest()
    (folder / f"cases-{digest[:12]}.json").write_text(manifest)
    # Only model-visible inputs cross to Modal. Rubrics/reference answers do not.
    inputs = [{k: row[k] for k in ("id", "messages", "tools") if k in row} for row in rows]
    print(f"Running {candidate}: {len(inputs)} cases; manifest {digest}", flush=True)
    runner = compare.with_options(gpu="A100-80GB", memory=49152) if candidate == "gemma31" else compare
    report = runner.remote(inputs, candidate, language_context, publisher_sampling)
    report["manifest_sha256"] = digest
    path = folder / f"{candidate}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for row in report["results"]:
        print(json.dumps({k: v for k, v in row.items() if k not in ("output_token_ids", "raw_output")}, ensure_ascii=False), flush=True)
    print(f"Saved {path}", flush=True)
