#!/usr/bin/env python3
"""Loopback-only, quantized R7/R6 comparison. No cloud client or fallback."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time
import uuid

os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", GRADIO_ANALYTICS_ENABLED="False")
from prepare_r7_local_mlx import BASE_SHA, MODEL, REVISION, VARIANTS, require, sha
from response_loop_guard import repeated_passage, VERSION as GUARD_VERSION

ORDER = ("r7", "r6")
MAX_TOKENS = 256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fresh():
    return {"id": uuid.uuid4().hex, "rounds": [], "history_policy": "own_complete_replies_only"}


def history(state, variant):
    require(variant in ORDER, "Unknown variant")
    result = []
    for turn in state["rounds"]:
        side = turn["sides"].get(variant, {})
        if side.get("complete") and side.get("answer", "").strip():
            result.extend([{"role": "user", "content": turn["question"]},
                           {"role": "assistant", "content": side["answer"]}])
    return result


def visible(state, variant):
    result = []
    for turn in state["rounds"]:
        result.append({"role": "user", "content": turn["question"]})
        answer = turn["sides"].get(variant, {}).get("answer")
        if answer:
            result.append({"role": "assistant", "content": answer})
    return result


class Runtime:
    def __init__(self, base, adapters, bundle):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import load_adapters
        from check_afrique_training_contract import load_tokenizer

        identity = json.loads((base / "source-identity.json").read_text())
        require(identity["source_sha256"] == BASE_SHA and identity["revision"] == REVISION, "Wrong quantized base")
        require(identity["quantization"] == {"bits": 4, "group_size": 64, "mode": "affine"}, "Wrong quantization")
        for name, checksum in identity["files"].items():
            require(Path(name).name == name and sha(base / name) == checksum, "Changed quantized artifact: " + name)
        self.adapter_manifest = json.loads((adapters / "manifest.json").read_text())
        require(self.adapter_manifest["base"] == MODEL and self.adapter_manifest["revision"] == REVISION, "Wrong adapter base")
        for variant in ORDER:
            record = self.adapter_manifest["variants"][variant]
            require(record["source_weights_sha256"] == VARIANTS[variant][1], "Wrong source adapter")
            require(sha(adapters / variant / "adapters.safetensors") == record["converted_weights_sha256"], "Changed adapter")
            require(sha(adapters / variant / "adapter_config.json") == record["converted_config_sha256"], "Changed adapter config")
        mx.set_memory_limit(8 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)
        self.model, _ = load(str(base))
        self.model = load_adapters(self.model, adapters / "r7")
        self.model.eval()
        self.tokenizer = load_tokenizer(bundle / "tokenizer")
        self.weights = {v: mx.load(str(adapters / v / "adapters.safetensors")) for v in ORDER}
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.identity = {"mode": "local-mlx-4bit", "base": MODEL, "base_revision": REVISION,
            "source_base_sha256": BASE_SHA, "quantized_files": identity["files"],
            "adapters": self.adapter_manifest["variants"],
            "runtime": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers")},
            "decoding": {"temperature": 0, "max_new_tokens": MAX_TOKENS, "enable_thinking": False,
                         "repetition_penalty": 1.0, "context_tokens": 4096, "truncate": False},
            "guard_version": GUARD_VERSION, "cloud_requests": False,
            "comparable_to_previous_bf16_scores": False}

    def stream(self, variant, messages, guarded=True):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        from afrique_instruction_baseline_core import prompt_ids

        require(variant in ORDER, "Unknown variant")
        ids = prompt_ids(self.tokenizer, messages, {"context_tokens": 4096, "max_new_tokens": MAX_TOKENS})
        record = {"variant": variant, "messages": copy.deepcopy(messages), "prompt_ids_sha256": digest(ids),
                  "answer": "", "output_ids": [], "complete": False, "finish_reason": "pending",
                  "guarded": guarded, "guard_trigger": None, "first_text_seconds": None}
        start = time.monotonic()
        with self.lock:
            self.model.load_weights(list(self.weights[variant].items()), strict=False)
            mx.eval(self.model.parameters())
            generator = stream_generate(self.model, self.tokenizer, prompt=ids, max_tokens=MAX_TOKENS,
                                        sampler=make_sampler(temp=0), prefill_step_size=256)
            try:
                for output in generator:
                    record["output_ids"].append(output.token)
                    record["answer"] += output.text
                    if output.text and record["first_text_seconds"] is None:
                        record["first_text_seconds"] = time.monotonic() - start
                    record["seconds"] = time.monotonic() - start
                    record["peak_memory_gb"] = output.peak_memory
                    record["guard_trigger"] = repeated_passage(record["answer"]) if guarded else None
                    if self.stop.is_set():
                        record["finish_reason"] = "cancelled"
                    elif record["guard_trigger"]:
                        record["finish_reason"] = "repetition_guard"
                    elif record["seconds"] > 180:
                        record["finish_reason"] = "time_limit"
                    elif output.finish_reason:
                        record["finish_reason"] = "eos" if output.finish_reason == "stop" else "max_new_tokens"
                        record["complete"] = output.finish_reason == "stop" and bool(record["answer"].strip())
                    yield copy.deepcopy(record)
                    if record["finish_reason"] != "pending":
                        break
            finally:
                generator.close()
                mx.clear_cache()


def serve(runtime, store, port):
    import gradio as gr
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity_sha = digest(runtime.identity)

    def persist(state):
        state["runtime_sha256"] = identity_sha
        path = store / (state["id"] + ".json")
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        partial.replace(path)

    def choices():
        records = [json.loads(p.read_text()) for p in sorted(store.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)]
        return [(r["rounds"][0]["question"][:55], r["id"]) for r in records
                if r.get("runtime_sha256") == identity_sha and r["rounds"]]

    def statuses(state):
        result = []
        for v in ORDER:
            row = (state["rounds"][-1]["sides"].get(v, {}) if state["rounds"] else {})
            reason = row.get("finish_reason", "ready")
            label = {"ready": "Ready", "pending": "Responding", "eos": "Complete",
                     "repetition_guard": "Paused", "cancelled": "Stopped", "error": "Unable to complete",
                     "time_limit": "Time limit", "max_new_tokens": "Length limit"}.get(reason, reason)
            if row.get("seconds") is not None:
                label += f" | {row['seconds']:.1f}s"
            if reason not in ("ready", "pending", "eos"):
                label += " | Partial turn excluded from context"
            result.append(label)
        return result

    def view(state):
        return [state, visible(state, "r7"), visible(state, "r6"), *statuses(state),
                gr.update(choices=choices(), value=state["id"] if state["rounds"] else None)]

    def compare(question, state, guarded):
        require(bool(question.strip()) and len(question) <= 16000, "Enter a message of at most 16000 characters")
        current = copy.deepcopy(state or fresh())
        require(len(current["id"]) == 32 and all(c in "0123456789abcdef" for c in current["id"]), "Invalid session")
        saved_path = store / (current["id"] + ".json")
        if saved_path.exists():
            current = json.loads(saved_path.read_text())
            require(current["runtime_sha256"] == identity_sha, "Session belongs to another runtime")
        from afrique_instruction_baseline_core import prompt_ids
        inputs = {v: history(current, v) + [{"role": "user", "content": question}] for v in ORDER}
        for messages in inputs.values():
            prompt_ids(runtime.tokenizer, messages, {"context_tokens": 4096, "max_new_tokens": MAX_TOKENS})
        runtime.stop.clear()
        turn = {"question": question, "sides": {}, "guarded": guarded}
        current["rounds"].append(turn)
        persist(current)
        yield view(current)
        for v in ORDER:
            if runtime.stop.is_set():
                turn["sides"][v] = {"finish_reason": "cancelled", "complete": False}
                continue
            try:
                for record in runtime.stream(v, inputs[v], guarded):
                    turn["sides"][v] = record
                    yield view(current)
            except Exception as error:
                turn["sides"].setdefault(v, {}).update(finish_reason="error", complete=False, error=str(error))
            finally:
                persist(current)
        yield view(current)

    def new_chat():
        return view(fresh())

    def restore(identifier):
        if not identifier:
            return view(fresh())
        require(len(identifier) == 32 and all(c in "0123456789abcdef" for c in identifier), "Invalid session")
        state = json.loads((store / (identifier + ".json")).read_text())
        require(state["runtime_sha256"] == identity_sha, "Session belongs to another runtime")
        return view(state)

    (store / "runtime.json").write_text(json.dumps(runtime.identity, indent=2))
    with gr.Blocks(title="Twi R7 Local", analytics_enabled=False) as app:
        gr.Markdown("# Twi R7 Local\n4-bit on this Mac. Research, not medical advice.")
        state = gr.State(fresh())
        with gr.Row():
            saved = gr.Dropdown(choices=choices(), label="Saved conversation", scale=4)
            guarded = gr.Checkbox(value=True, label="Repetition guard")
            new = gr.Button("New conversation")
        with gr.Row():
            with gr.Column(min_width=300):
                left = gr.Chatbot(label="R7", height=480, sanitize_html=True)
                left_status = gr.Markdown("Ready")
            with gr.Column(min_width=300):
                right = gr.Chatbot(label="R6", height=480, sanitize_html=True)
                right_status = gr.Markdown("Ready")
        with gr.Row():
            prompt = gr.Textbox(label="Message", placeholder="Twi or English", scale=8)
            send = gr.Button("Send", variant="primary")
            stop = gr.Button("Stop")
        with gr.Accordion("Model identity", open=False):
            gr.JSON(runtime.identity)
        outputs = [state, left, right, left_status, right_status, saved]
        event = gr.on([send.click, prompt.submit], compare, [prompt, state, guarded], outputs,
                      concurrency_limit=1, concurrency_id="model", api_name="compare")
        stop.click(runtime.stop.set, queue=False, api_name=False)
        new.click(new_chat, outputs=outputs, concurrency_id="model", api_name=False)
        saved.input(restore, saved, outputs, concurrency_id="model", api_name=False)
    app.queue(max_size=2).launch(server_name="127.0.0.1", server_port=port, share=False,
                                theme=gr.themes.Base(primary_hue="gray"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "serve"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--prompt", default="wo ho te s3n?")
    args = parser.parse_args()
    runtime = Runtime(args.base, args.adapters, args.bundle)
    if args.command == "serve":
        serve(runtime, args.store, args.port)
    else:
        args.store.mkdir(parents=True, exist_ok=True)
        for variant in ORDER:
            for record in runtime.stream(variant, [{"role": "user", "content": args.prompt}]):
                pass
            destination = args.store / (variant + "-" + uuid.uuid4().hex + ".json")
            destination.write_text(json.dumps({"identity": runtime.identity, "result": record}, ensure_ascii=False, indent=2))
            print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
