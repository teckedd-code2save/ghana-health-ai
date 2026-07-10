"""
Fine-tune / SFT the understanding model for Twi–English health chat.

  1. Build JSONL: {"messages":[{"role":"system","content":...},{"role":"user"...},{"role":"assistant"...}]}
  2. Push as HF dataset or mount on Modal volume
  3. Run LoRA SFT on Llama-3.1-8B or Qwen2.5-7B

  modal run modal/train/train_understand.py --smoke
  modal run modal/train/train_understand.py \\
    --dataset teckedd/gha-health-sft-v1 \\
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --max-steps 500 \\
    --push-repo teckedd/gha-understand-twi-v1

Until a dataset exists, this prints the data contract and exits cleanly.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import modal

app = modal.App("ghana-health-understand-train")
vol = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "peft==0.13.2",
        "trl==0.12.1",
        "bitsandbytes==0.44.1",
        "huggingface_hub==0.26.2",
    )
)

try:
    SECRETS = [modal.Secret.from_name("huggingface")]
except Exception:  # noqa: BLE001
    SECRETS = []


DATA_CONTRACT = {
    "format": "chat messages JSONL or HF dataset with `messages` column",
    "example": {
        "messages": [
            {
                "role": "system",
                "content": "You are Ghana Health AI. Reply in the user's language. Never use bare CHW.",
            },
            {"role": "user", "content": "Me ti yɛ me ya na me wɔ nyinsen"},
            {
                "role": "assistant",
                "content": (
                    "Mede asɛm no ate. Ti yaw wɔ nyinsen mu betumi ayɛ den — "
                    "sɛ ɛyɛ den anaa wowɔ fever a, kɔ clinic anaa community health worker hɔ ntɛm."
                ),
            },
        ],
        "intent": "HEALTH",
        "language": "tw",
        "severity": "MEDIUM",
    },
    "min_rows_for_lora": 500,
    "target_rows": 2000,
    "eval": "100 fixed scenarios: danger, market, nonsense ASR, mixed Twi-EN",
}


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    volumes={"/data": vol},
    secrets=SECRETS,
)
def train(
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    dataset_name: str = "",
    max_steps: int = 500,
    learning_rate: float = 2e-4,
    push_repo: Optional[str] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    if not dataset_name:
        return {
            "status": "awaiting_data",
            "message": "Build SFT dataset first (GAIN cleaned + clinician filter).",
            "data_contract": DATA_CONTRACT,
            "base_model": base_model,
        }

    # Full TRL SFTTrainer loop — enabled when dataset is present
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = "/data/hf"
    out_dir = f"/data/sft/{base_model.replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    if smoke:
        max_steps = min(max_steps, 10)

    ds = load_dataset(dataset_name, token=token, cache_dir=cache)
    split = "train" if "train" in ds else list(ds.keys())[0]
    train_ds = ds[split]
    if smoke:
        train_ds = train_ds.select(range(min(8, len(train_ds))))

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, cache_dir=cache)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=token,
        cache_dir=cache,
        torch_dtype="auto",
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = SFTConfig(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        logging_steps=5,
        save_steps=max(50, max_steps // 5),
        bf16=True,
        push_to_hub=bool(push_repo) and not smoke,
        hub_model_id=push_repo,
        hub_token=token,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    vol.commit()

    return {
        "status": "ok",
        "output_dir": out_dir,
        "base_model": base_model,
        "n_train": len(train_ds),
        "push_repo": push_repo,
        "max_steps": max_steps,
    }


@app.local_entrypoint()
def main(
    dataset: str = "",
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    smoke: bool = False,
    push_repo: str = "",
    max_steps: int = 500,
):
    print(
        train.remote(
            dataset_name=dataset,
            base_model=base_model,
            smoke=smoke,
            push_repo=push_repo or None,
            max_steps=max_steps,
        )
    )
