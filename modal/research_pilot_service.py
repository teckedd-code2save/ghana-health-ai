"""Private Modal inference for the saved pilot; no public web endpoint."""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "ghana-understanding-pilot-private"
BASE = "ghananlpcommunity/MiniCPM5-1B-Twi"
REVISION = "d807ca1a3323972afafabff8f9affe2639e37b5c"
ADAPTER = "/data/sft/annotated_response_pilot_v1_20260908T213218Z"
SHA256 = "294fd2815664c0b3140927dc637e5df35fc43563eaa9d8c5bb48f3678c2762de"
app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.7.1", "transformers==5.3.0", "accelerate==1.10.1",
    "peft==0.18.0", "huggingface_hub==1.3.0", "safetensors==0.6.2",
)
volume = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)


def validate_messages(messages: list[dict], variant: str) -> None:
    if variant not in ("pilot", "base"):
        raise ValueError("Unknown model")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 31:
        raise ValueError("Start a new conversation after 15 replies")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Conversation roles must alternate")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 8000:
            raise ValueError("Each message must contain 1-8000 characters")
    if messages[-1]["role"] != "user" or sum(len(m["content"]) for m in messages) > 24000:
        raise ValueError("Conversation is too long; start a new one")


@app.cls(image=image, gpu="L4", cpu=2, memory=8192, timeout=180,
         startup_timeout=180, max_containers=1, min_containers=0,
         scaledown_window=60, retries=0, volumes={"/data": volume.read_only()})
class Pilot:
    @modal.enter()
    def setup(self):
        self.model = None

    def load(self):
        import hashlib
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.model is not None:
            return
        if hashlib.sha256(Path(ADAPTER, "adapter_model.safetensors").read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Adapter integrity check failed")
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            BASE, revision=REVISION, cache_dir="/data/hf", local_files_only=True,
            dtype=torch.bfloat16,
        ).to("cuda")
        self.model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False).eval()

    @modal.method(is_generator=True)
    def stream(self, messages: list[dict], variant: str = "pilot"):
        import time
        from contextlib import nullcontext
        from queue import Empty
        from threading import Event, Thread

        import torch
        from transformers import StoppingCriteria, TextIteratorStreamer, set_seed

        validate_messages(messages, variant)
        yield {"type": "started", "call_id": modal.current_function_call_id(),
               "model": BASE, "revision": REVISION, "variant": variant,
               "adapter_sha256": SHA256 if variant == "pilot" else None}
        self.load()
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        if inputs["input_ids"].shape[-1] > 3072:
            raise ValueError("Conversation exceeds the pilot context budget; start a new one")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1)
        stop, done, errors, results = Event(), Event(), [], []
        started = time.monotonic()

        class Halt(StoppingCriteria):
            def __call__(self, *args, **kwargs):
                return stop.is_set() or time.monotonic() - started > 90

        def generate():
            try:
                set_seed(42)
                with (nullcontext() if variant == "pilot" else self.model.disable_adapter()), torch.inference_mode():
                    results.append(self.model.generate(
                        **inputs, streamer=streamer, max_new_tokens=384, do_sample=True,
                        temperature=0.7, top_p=0.9, repetition_penalty=1.3,
                        no_repeat_ngram_size=3, stopping_criteria=[Halt()],
                        pad_token_id=self.tokenizer.pad_token_id,
                    ))
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        worker = Thread(target=generate, daemon=True)
        worker.start()
        try:
            while True:
                try:
                    chunk = next(streamer)
                    if chunk:
                        yield {"type": "delta", "text": chunk}
                except Empty:
                    if done.is_set():
                        break
                except StopIteration:
                    break
            worker.join(timeout=5)
            if errors:
                raise RuntimeError("Model generation failed") from errors[0]
            generated = results[0][0, inputs["input_ids"].shape[-1]:].tolist()
            reason = "stop" if generated and generated[-1] == self.tokenizer.eos_token_id else "limit"
            yield {"type": "done", "seconds": round(time.monotonic() - started, 3),
                   "output_tokens": len(generated), "finish_reason": reason}
        finally:
            stop.set()
            worker.join(timeout=5)
