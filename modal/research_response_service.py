"""Private, checksum-bound response inference; no public HTTP endpoint."""
from pathlib import Path

import modal

APP_NAME = "ghana-response-v2-private"
app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.10.0", "transformers==5.17.0", "peft==0.20.0", "accelerate==1.15.0", "jsonschema==4.26.0",
).env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1"})
if modal.is_local():
    image = image.add_local_file(str(Path(__file__).with_name("response_runtime.py")), "/root/response_runtime.py")
cache = modal.Volume.from_name("ghana-health-understanding-hf-cache", create_if_missing=False)
artifacts = modal.Volume.from_name("ghana-health-understand-train", create_if_missing=False)


@app.function(image=image, cpu=1, memory=2048, timeout=120, retries=0,
              volumes={"/artifacts": artifacts.read_only()})
def inspect_checkpoint(run_id: str, checkpoint: str):
    import hashlib
    import json
    import sys
    sys.path.insert(0, "/root")
    from response_runtime import model_spec, validate_input
    spec = model_spec(run_id)
    validate_input([{"role": "user", "content": "inspect"}], spec["variants"][0], run_id, checkpoint, "0" * 64)
    artifacts.reload()
    folder = Path("/artifacts", spec["folder"], run_id)
    with (folder / checkpoint / "adapter_model.safetensors").open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"run_id": run_id, "checkpoint": checkpoint, "adapter_sha256": sha,
            "training_status": json.loads((folder / "status.json").read_text()),
            "production_promoted": False}


@app.cls(image=image, gpu="H100", cpu=4, memory=49152, timeout=720, startup_timeout=240,
         max_containers=1, min_containers=0, scaledown_window=60, retries=0,
         volumes={"/cache": cache.read_only(), "/artifacts": artifacts.read_only()})
class Response:
    run_id: str = modal.parameter()
    checkpoint: str = modal.parameter()
    adapter_sha256: str = modal.parameter()

    @modal.enter()
    def setup(self):
        self.model = None

    @modal.method()
    def check(self, cases: list[dict], variant: str, profile: str = "greedy", seed: int = 42):
        if not isinstance(cases, list) or not 1 <= len(cases) <= 4:
            raise ValueError("A diagnostic batch contains at most four cases")
        output = []
        for case in cases:
            events, error = [], None
            try:
                for event in self.stream.local(case["messages"], variant, profile, seed):
                    events.append(event)
            except Exception as exc:
                error = type(exc).__name__
            output.append({"id": case["id"], "messages": case["messages"], "events": events, "error_type": error,
                           "answer": "".join(event["text"] for event in events if event["type"] == "delta")})
        return output

    def load(self):
        import hashlib
        import sys
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Gemma4ForCausalLM, Qwen3_5ForCausalLM
        sys.path.insert(0, "/root")
        from response_runtime import model_spec, QWEN_RESPONSE_TEMPLATE

        if self.model is not None:
            return
        artifacts.reload()
        spec = model_spec(self.run_id)
        path = Path("/artifacts", spec["folder"], self.run_id, self.checkpoint)
        with (path / "adapter_model.safetensors").open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != self.adapter_sha256:
                raise ValueError("Adapter integrity check failed")
        self.tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"], revision=spec["tokenizer_revision"], cache_dir=spec.get("tokenizer_cache_dir", spec["cache_dir"]), local_files_only=True)
        if spec["family"] == "qwen":
            self.tokenizer.response_template = QWEN_RESPONSE_TEMPLATE
        model_class = Gemma4ForCausalLM if spec["family"] == "gemma" else Qwen3_5ForCausalLM
        base, info = model_class.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.bfloat16,
            device_map={"": "cuda"}, attn_implementation="sdpa", output_loading_info=True,
            key_mapping={r"^model\.language_model\.": "model."}, cache_dir=spec["cache_dir"], local_files_only=True)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError("Base weights did not load completely")
        self.model = PeftModel.from_pretrained(base, path, is_trainable=False).eval()

    @modal.method(is_generator=True)
    def stream(self, messages: list[dict], variant: str = "response_v2", profile: str = "publisher", seed: int = 42):
        import sys
        import time
        from contextlib import nullcontext
        from queue import Empty
        from threading import Event, Thread

        import torch
        from transformers import StoppingCriteria, TextIteratorStreamer, set_seed

        sys.path.insert(0, "/root")
        from response_runtime import SYSTEM, TOOLS, complete_tool_handoff, content_chunks, decoding_profile, execute_calls, model_spec, presence_processor, validate_input

        validate_input(messages, variant, self.run_id, self.checkpoint, self.adapter_sha256)
        spec = model_spec(self.run_id)
        enabled = variant == spec["variants"][0]
        if not isinstance(seed, int) or not 0 <= seed <= 1000:
            raise ValueError("Invalid decoding seed")
        settings = decoding_profile(spec["family"], profile)
        presence_penalty, decoding = settings["presence_penalty"], settings["generation"]
        yield {"type": "started", "call_id": modal.current_function_call_id(), "model": spec["model"],
               "revision": spec["revision"], "variant": variant, "run_id": self.run_id, "checkpoint": self.checkpoint,
               "adapter_sha256": self.adapter_sha256 if enabled else None,
               "decoding": {**decoding, **{k: v for k, v in settings.items() if k != "generation"}}, "seed": seed,
               "connected_tools": ["calculate_total"], "proprietary_fallback": False}
        self.load()
        endings = [1, 106, 50] if spec["family"] == "gemma" else [self.tokenizer.eos_token_id]
        special = set(self.tokenizer.all_special_ids)
        if any("<|" in m["content"] or any(t in m["content"] for t in ("<think>", "</think>", "<tool_call>", "</tool_call>"))
               or special.intersection(self.tokenizer.encode(m["content"], add_special_tokens=False)) for m in messages):
            raise ValueError("Conversation contains model control tokens")
        history = [{"role": "system", "content": SYSTEM}, *messages]
        started, total_tokens = time.monotonic(), 0
        for turn in range(3):
            prompt = self.tokenizer.apply_chat_template(history, tools=TOOLS, tokenize=False,
                         add_generation_prompt=True, enable_thinking=settings["thinking"])
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
            if inputs["input_ids"].shape[1] > 4096:
                raise ValueError("Conversation exceeds context budget; no silent truncation")
            parser = self.tokenizer.get_response_parser(prefix=prompt, tools=TOOLS)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False, timeout=1)
            stop, done, errors, results = Event(), Event(), [], []

            class Halt(StoppingCriteria):
                def __call__(self, *args, **kwargs):
                    return stop.is_set() or time.monotonic() - started > 150

            def generate():
                try:
                    set_seed(seed)
                    with (nullcontext() if enabled else self.model.disable_adapter()), torch.inference_mode():
                        results.append(self.model.generate(**inputs, streamer=streamer, max_new_tokens=settings["max_new_tokens"],
                            **decoding, eos_token_id=endings, pad_token_id=self.tokenizer.pad_token_id,
                            logits_processor=[presence_processor(inputs["input_ids"].shape[1], presence_penalty)] if presence_penalty else None,
                            stopping_criteria=[Halt()]))
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
                        for text in content_chunks(parser.feed(chunk)):
                            yield {"type": "delta", "text": text}
                    except Empty:
                        if done.is_set():
                            break
                    except StopIteration:
                        break
                worker.join(timeout=5)
                if errors or not results:
                    raise RuntimeError("Generation failed") from (errors[0] if errors else None)
                parsed, events = parser.finalize()
                for text in content_chunks(events):
                    yield {"type": "delta", "text": text}
                ids = results[0][0, inputs["input_ids"].shape[1]:].tolist()
                total_tokens += len(ids)
                ended = bool(ids and ids[-1] in endings)
                raw = self.tokenizer.decode(ids, skip_special_tokens=False)
                yield {"type": "trace", "turn": turn, "raw_output": raw,
                       "parsed": parsed, "finish_reason": "stop" if ended else "limit"}
                if parsed.get("tool_calls"):
                    if not complete_tool_handoff(raw, ids, parsed["tool_calls"], spec["family"]) or turn == 2 or time.monotonic() - started > 150:
                        raise ValueError("Tool generation exceeded its budget; no action executed")
                    calls, replies = execute_calls(parsed["tool_calls"], messages)
                    assistant = {"role": "assistant", "content": parsed.get("content", ""), "tool_calls": calls}
                    if settings["thinking"] and parsed.get("thinking"):
                        assistant["reasoning"] = parsed["thinking"]
                    history.append(assistant)
                    history.extend(replies)
                    yield {"type": "tool_result", "calls": calls, "results": replies}
                else:
                    if not parsed.get("content", "").strip():
                        raise ValueError("Model returned no answer")
                    yield {"type": "done", "seconds": round(time.monotonic() - started, 3),
                           "output_tokens": total_tokens, "finish_reason": "stop" if ended else "limit"}
                    return
            finally:
                stop.set()
                worker.join(timeout=10)
                if worker.is_alive():
                    raise RuntimeError("Generation worker did not stop")
