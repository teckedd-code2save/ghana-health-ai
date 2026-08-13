"""
Contrastive embedding train for noisy Twi ASR recovery — go for gold.

If --dataset is empty, generates synthetic clean↔noisy pairs from a built-in
Twi health/market bank (same meanings as product semantic bank), trains
MultipleNegativesRankingLoss on multilingual-e5-base, and optionally pushes.

  modal run --detach modal/train/train_embeddings.py \\
    --run-name v1-synth --max-steps 1200 \\
    --push-repo teckedd/gha-embed-twi-health-v1

  modal run modal/train/train_embeddings.py --smoke
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, Optional

import modal

app = modal.App("ghana-health-embed-train")
vol = modal.Volume.from_name("ghana-health-embed-train", create_if_missing=True)

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "sentence-transformers==3.3.1",
        "huggingface_hub==0.26.2",
        "numpy<2.3",
    )
    .add_local_file(
        local_path=os.path.join(_TRAIN_DIR, "model_card.py"),
        remote_path="/root/gha_train/model_card.py",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


# Built-in anchors — Twi-first health + market (expand over time)
CLEAN_BANK: list[dict[str, str]] = [
    {"text": "mogya resen dodo me yafunu mu", "meaning": "heavy_bleeding_pregnancy"},
    {"text": "mente me ba a ɔwɔ me yafunu mu bio", "meaning": "reduced_fetal_movement"},
    {"text": "me ho repopo anaa me nte yie koraa", "meaning": "seizure_or_collapse"},
    {"text": "me wɔ afe na me ho yɛ hyew", "meaning": "fever"},
    {"text": "me wɔ nyinsen mu na me wɔ afe", "meaning": "fever_in_pregnancy"},
    {"text": "me ti yɛ me ya", "meaning": "headache"},
    {"text": "me ti yɛ me ya na me wɔ nyinsen", "meaning": "headache_pregnancy"},
    {"text": "me pɛ sɛ mekɔ antenatal care", "meaning": "anc_visit"},
    {"text": "aduane bɛn na ɛsɛ sɛ midi wɔ nyinsen mu", "meaning": "pregnancy_nutrition"},
    {"text": "mawo na mogya da so resen", "meaning": "postpartum_bleeding"},
    {"text": "ɛmo boɔ yɛ sɛn", "meaning": "price_rice"},
    {"text": "me pɛ paracetamol", "meaning": "buy_paracetamol"},
    {"text": "fa ka me cart ho", "meaning": "add_to_cart"},
    {"text": "wo ho te sɛn", "meaning": "greeting"},
    {"text": "medaase", "meaning": "thanks"},
    {"text": "bere bɛn na ɛsɛ sɛ mekɔ clinic", "meaning": "when_to_clinic"},
    {"text": "me yafunu yɛ me ya", "meaning": "abdominal_pain"},
    {"text": "me nufu yɛ me ya", "meaning": "breast_pain"},
    {"text": "me ba nnom nufu yie", "meaning": "breastfeeding_trouble"},
    {"text": "me tutu na me ho yɛ mmerɛw", "meaning": "diarrhea_weak"},
    {"text": "me fe na me nte yie", "meaning": "vomiting"},
    {"text": "ahonhon wɔ me nsa ne me anim", "meaning": "swelling_danger"},
    {"text": "me nte home yie", "meaning": "breathing_difficulty"},
    {"text": "me bo yɛ me ya", "meaning": "chest_pain"},
    {"text": "me pɛ sɛ metɔ sapo", "meaning": "buy_soap"},
    {"text": "ngo boɔ yɛ sɛn", "meaning": "price_oil"},
    {"text": "me wɔ malaria", "meaning": "malaria"},
    {"text": "me pɛ ORS", "meaning": "buy_ors"},
    {"text": "sɛn na me yɛ sɛ me wɔ afe", "meaning": "fever_advice"},
    {"text": "me ho yɛ me anika wɔ nyinsen mu", "meaning": "pregnancy_worry"},
    {"text": "I am bleeding heavily and I am pregnant", "meaning": "heavy_bleeding_pregnancy"},
    {"text": "I have high fever and chills", "meaning": "fever"},
    {"text": "how much is cooking oil", "meaning": "price_oil"},
]


def _noise(text: str, rng: random.Random) -> str:
    """Simulate ASR / dialect / orthography mess for Twi."""
    t = text
    ops = [
        lambda s: s.replace("ɛ", "e").replace("ɔ", "o"),
        lambda s: s.replace("ɛ", "e"),
        lambda s: s.replace("ɔ", "o"),
        lambda s: re.sub(r"\s+", " ", s),
        lambda s: s.lower(),
        lambda s: re.sub(r"([aeiouɛɔ])\1+", r"\1", s),
        lambda s: " ".join(s.split()[: max(1, len(s.split()) - 1)]) if len(s.split()) > 3 else s,
        lambda s: s.replace("yɛ", "ye").replace("wɔ", "wo"),
        lambda s: s.replace("nyinsen", "nyin sen").replace("mogya", "mo gya"),
        lambda s: s.replace("afe", "afe").replace("clinic", "klinik"),
        lambda s: "".join(c for i, c in enumerate(s) if not (c == " " and rng.random() < 0.08)),
        lambda s: s + (" " + rng.choice(["aaa", "eii", "hmm"]) if rng.random() < 0.15 else ""),
    ]
    n_ops = rng.randint(1, 3)
    for _ in range(n_ops):
        t = rng.choice(ops)(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t and t != text else text.lower().replace("ɛ", "e")


def build_synth_pairs(n: int = 4000, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []
    for _ in range(n):
        row = rng.choice(CLEAN_BANK)
        anchor = row["text"]
        positive = _noise(anchor, rng)
        # same-meaning paraphrase sometimes: use another bank line with same meaning
        same = [r["text"] for r in CLEAN_BANK if r["meaning"] == row["meaning"]]
        if len(same) > 1 and rng.random() < 0.25:
            positive = rng.choice([s for s in same if s != anchor] or [positive])
        pairs.append(
            {
                "anchor": anchor,
                "positive": positive,
                "meaning": row["meaning"],
                "language": "tw" if any(c in anchor for c in "ɛɔ") or " me " in f" {anchor} " else "en",
            }
        )
    return pairs


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    volumes={"/data": vol},
    secrets=SECRETS,
)
def train(
    base_model: str = "intfloat/multilingual-e5-base",
    dataset_name: str = "",
    max_steps: int = 1200,
    learning_rate: float = 2e-5,
    push_repo: Optional[str] = None,
    smoke: bool = False,
    run_name: str = "v1",
    n_synth: int = 4000,
) -> dict[str, Any]:
    import sys

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/data/hf"
    out_dir = f"/data/embed_runs/{run_name}_{base_model.replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    if smoke:
        max_steps = min(max_steps, 30)
        n_synth = min(n_synth, 64)

    rows: list[dict[str, str]] = []
    source = "synth_bank"
    if dataset_name:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, token=token, cache_dir=cache)
        split = "train" if "train" in ds else list(ds.keys())[0]
        train_ds = ds[split]
        for row in train_ds:
            a = str(row.get("anchor") or "").strip()
            p = str(row.get("positive") or "").strip()
            if a and p:
                rows.append({"anchor": a, "positive": p, "meaning": str(row.get("meaning") or "")})
        source = dataset_name
    else:
        rows = build_synth_pairs(n=n_synth)

    if len(rows) < 8:
        return {"status": "error", "message": "Too few pairs", "n": len(rows)}

    model = SentenceTransformer(base_model, cache_folder=cache)
    examples = [InputExample(texts=[r["anchor"], r["positive"]]) for r in rows]

    batch_size = 16 if not smoke else 4
    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # steps_per_epoch ≈ len / batch; fit epochs so we roughly hit max_steps
    steps_per_epoch = max(1, len(examples) // batch_size)
    epochs = max(1, min(8, (max_steps + steps_per_epoch - 1) // steps_per_epoch))

    print(
        f"[embed-train] source={source} n={len(examples)} "
        f"epochs={epochs} steps/epoch≈{steps_per_epoch} max_steps={max_steps}"
    )

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=min(100, max(1, max_steps // 10)),
        output_path=out_dir,
        show_progress_bar=True,
        optimizer_params={"lr": learning_rate},
    )
    vol.commit()

    # Quick in-batch retrieval sanity: noisy query → clean anchor same meaning
    import torch
    from sentence_transformers import util

    eval_n = min(40, len(CLEAN_BANK))
    q_texts = [_noise(CLEAN_BANK[i]["text"], random.Random(i + 7)) for i in range(eval_n)]
    c_texts = [CLEAN_BANK[i]["text"] for i in range(eval_n)]
    q_emb = model.encode(q_texts, normalize_embeddings=True, convert_to_tensor=True)
    c_emb = model.encode(c_texts, normalize_embeddings=True, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, c_emb, top_k=1)
    r1 = sum(1 for i, h in enumerate(hits) if h and h[0]["corpus_id"] == i) / eval_n

    hub_status = None
    if push_repo and not smoke and token:
        try:
            model.save_to_hub(push_repo, token=token, exist_ok=True)
            sys.path.insert(0, "/root/gha_train")
            from model_card import write_and_push_model_card  # type: ignore

            write_and_push_model_card(
                push_repo,
                task="feature-extraction",
                language=["tw", "ak", "en"],
                base_model=base_model,
                datasets=[source],
                metrics={
                    "n_pairs": len(examples),
                    "retrieval_at_1_synth_noise": float(r1),
                    "max_steps": max_steps,
                },
                summary=(
                    f"Twi-primary health embedding ({run_name}) for Ghana Health AI. "
                    f"Noise-robust retrieval@1={r1:.3f} on synthetic ASR-like probes."
                ),
                tags=["sentence-transformers", "embeddings", "twi", "asr-robust"],
                pipeline_tag="feature-extraction",
                token=token,
            )
            hub_status = f"pushed:{push_repo}+card"
        except Exception as exc:  # noqa: BLE001
            hub_status = f"push_failed:{exc}"
            print(f"[embed-train] hub: {exc}")

    summary = {
        "status": "ok",
        "run_name": run_name,
        "output_dir": out_dir,
        "base_model": base_model,
        "source": source,
        "n_pairs": len(examples),
        "retrieval_at_1_synth_noise": float(r1),
        "push_repo": push_repo,
        "hub": hub_status,
        "serve": "Set EMBED_MODEL_ID on modal/embed_service.py after promote",
    }
    print("[embed-train] done", summary)
    return summary


@app.local_entrypoint()
def main(
    dataset: str = "",
    smoke: bool = False,
    push_repo: str = "teckedd/gha-embed-twi-health-v1",
    max_steps: int = 1200,
    base_model: str = "intfloat/multilingual-e5-base",
    run_name: str = "v1-synth",
    n_synth: int = 4000,
    no_push: bool = False,
):
    print(
        train.remote(
            dataset_name=dataset,
            smoke=smoke,
            push_repo=None if no_push or smoke else (push_repo or None),
            max_steps=max_steps,
            base_model=base_model,
            run_name=run_name,
            n_synth=n_synth,
        )
    )
