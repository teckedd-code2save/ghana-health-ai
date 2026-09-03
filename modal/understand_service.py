"""
Ghana Health AI — Twi semantic recovery service on Modal.

Loads the research LoRA adapter:
  base: Qwen/Qwen2.5-1.5B-Instruct
  adapter: teckedd/gha-understand-twi-medical-silver-v1

This endpoint recovers structured meaning only. Product medical replies still go
through the application response/safety pipeline.

  modal deploy modal/understand_service.py
"""

# NOTE: Do not add `from __future__ import annotations` — keeps FastAPI/Modal
# runtime annotation handling simple.
import json
import os
import time
from typing import Any, Optional

import modal

APP_NAME = os.environ.get("UNDERSTAND_APP_NAME", "ghana-health-understand")
BASE_MODEL = os.environ.get("UNDERSTAND_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_ID = os.environ.get(
    "UNDERSTAND_ADAPTER_ID", "teckedd/gha-understand-twi-medical-silver-v1"
)
MAX_NEW_TOKENS = int(os.environ.get("UNDERSTAND_MAX_NEW_TOKENS", "320"))

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("ghana-health-understand-models", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "UNDERSTAND_BASE_MODEL": BASE_MODEL,
            "UNDERSTAND_ADAPTER_ID": ADAPTER_ID,
            "UNDERSTAND_SERVICE_NAME": APP_NAME,
        }
    )
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "peft==0.17.1",
        "huggingface_hub==0.26.2",
    )
)

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]==0.115.12"
)

try:
    SECRETS = [modal.Secret.from_name("huggingface-token")]
except Exception:  # noqa: BLE001
    SECRETS = []


SYSTEM = (
    "You are Ghana Health AI's semantic recovery model. Given a Twi/Akan, "
    "English, or code-switched user utterance, output faithful structured "
    "understanding. Do not diagnose. Do not invent missing symptoms. Preserve "
    "uncertainty. Return JSON only with keys normalized_twi, natural_english, "
    "literal_english, intent, entities, ambiguities, requires_clarification."
)


def _json_from_text(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no_json_object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("json_not_object")
    return value


def _fallback_payload(text: str, model: str, started: float, error: str) -> dict[str, Any]:
    return {
        "normalized_twi": text.strip(),
        "natural_english": "",
        "literal_english": "",
        "intent": "unknown",
        "entities": {},
        "ambiguities": f"service_error={error}",
        "requires_clarification": True,
        "model": model,
        "latency_ms": int((time.time() - started) * 1000),
        "error": error,
    }


@app.cls(
    image=gpu_image,
    gpu="T4",
    timeout=240,
    scaledown_window=60,
    max_containers=1,
    volumes={"/models": model_volume},
    secrets=SECRETS,
)
class UnderstandEngine:
    @modal.enter()
    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        cache = "/models/hf"
        self.base_model = os.environ.get("UNDERSTAND_BASE_MODEL", BASE_MODEL)
        self.adapter_id = os.environ.get("UNDERSTAND_ADAPTER_ID", ADAPTER_ID)
        self.model_id = self.adapter_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs: dict[str, Any] = {"cache_dir": cache}
        if token:
            kwargs["token"] = token

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_id, **kwargs)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            cache_dir=cache,
            torch_dtype="auto",
            device_map="auto",
            token=token,
        )
        self.model = PeftModel.from_pretrained(
            base,
            self.adapter_id,
            cache_dir=cache,
            token=token,
        )
        self.model.eval()
        print(
            f"[understand] ready base={self.base_model} adapter={self.adapter_id} "
            f"device={self.device}"
        )

    @modal.method()
    def predict(
        self,
        text: str,
        language: str = "tw",
        focus: str = "health",
        history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        import torch

        started = time.time()
        text = (text or "").strip()
        if not text:
            return _fallback_payload(text, self.model_id, started, "empty_text")

        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f"language={language}\nfocus={focus}\n"
                    f"recent_history={json.dumps((history or [])[-4:], ensure_ascii=False)}\n"
                    f"utterance={text}"
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][encoded["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            payload = _json_from_text(decoded)
        except Exception as exc:  # noqa: BLE001
            return _fallback_payload(text, self.model_id, started, f"parse_failed:{exc}")

        payload.setdefault("normalized_twi", text)
        payload.setdefault("natural_english", "")
        payload.setdefault("literal_english", "")
        payload.setdefault("intent", "unknown")
        payload.setdefault("entities", {})
        payload.setdefault("ambiguities", "")
        payload.setdefault("requires_clarification", False)
        payload["model"] = self.model_id
        payload["base_model"] = self.base_model
        payload["latency_ms"] = int((time.time() - started) * 1000)
        return payload


@app.function(image=web_image)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    api_app = FastAPI(title="Ghana Health AI Understanding")
    engine = UnderstandEngine()

    class UnderstandRequest(BaseModel):
        text: str
        language: str = "tw"
        focus: str = "health"
        history: list[dict[str, str]] = []
        memory: Any = None
        transcript: Any = None

    def _check_auth(authorization: Optional[str]) -> None:
        expected = os.environ.get("UNDERSTAND_API_TOKEN")
        if expected and authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    @api_app.get("/health")
    async def health() -> dict[str, str]:
        return {"ok": "true", "model": ADAPTER_ID}

    @api_app.post("/understand")
    async def understand(
        body: UnderstandRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)
        return engine.predict.remote(
            text=body.text,
            language=body.language,
            focus=body.focus,
            history=body.history,
        )

    return api_app
