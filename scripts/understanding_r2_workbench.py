#!/usr/bin/env python3
"""Loopback-only paired R2 evaluation UI; preserves the existing chat store."""
import argparse
import html
import os
from pathlib import Path
import threading

import understanding_r2_demo as demo
import r2_comparison as comparison

LABELS = {"r2_meaning": "R2: language + meaning (current)",
          "r2_language": "R2: language stage", "r2_response": "R2: final response stage",
          "r1_meaning": "R1: meaning", "r1_response": "R1: final response stage",
          "r3_response": "R3: final response stage"}
CSS = demo.CSS + """
.gradio-container { max-width:1440px !important; }
#compare-panes { gap:24px !important; }
.compare-chat { border:0 !important; background:transparent !important; }
.compare-chat .message { border-radius:8px !important; box-shadow:none !important; }
.compare-chat .message p { overflow-wrap:anywhere; line-height:1.65; }
.compare-chat button[aria-label="Clear"] { display:none !important; }
.evidence-strip { min-height:58px; font-size:13px; color:#566170; }
.evidence-strip strong { color:#202a35; }
#comparison-status { min-height:30px; font-size:13px; }
#lab-notice { font-size:13px; color:#566170; }
#compare-panes h3 { font-size:18px; margin:6px 0; }
#compare-panes code { white-space:normal; overflow-wrap:anywhere; font-size:11px; }
@media(max-width:700px) { .gradio-container { padding:14px !important; } #compare-panes { gap:12px !important; } }
"""


def transcript(state, side):
    rows = []
    for item in state.get("rounds", []):
        rows.append({"role": "user", "content": item["question"]})
        turn = item["sides"][side].get("turn")
        if turn and turn.get("answer"):
            rows.append({"role": "assistant", "content": turn["answer"]})
    return rows


def candidate_heading(candidate, manifest=None):
    name, stage = LABELS[candidate].split(": ", 1)
    if manifest and manifest.get("version") == "r3-comparison-inference-v1":
        stage = stage.removesuffix(" (current)")
    text = f"### {name}\n{stage}"
    if manifest is not None:
        checksum = manifest["variants"][candidate]["files"]["adapter_model.safetensors"]
        text += f"\n\n`{candidate}`\n\n`{checksum}`"
    return text


def side_metrics(side, previous_answer=None):
    if not side:
        return "Awaiting comparison"
    turn = side.get("turn") or {}
    result = turn.get("remote_result") or {}
    first, wall = side.get("ttft_seconds"), side.get("wall_seconds")
    fields = ["Status: " + str(side["status"])]
    if first is not None:
        fields.append(f"First text: {first:.2f}s")
    if wall is not None:
        fields.append(f"Elapsed: {wall:.2f}s")
    if result:
        fields.extend(["Stop: " + result["finish_reason"], f"Output tokens: {len(result['output_ids'])}"])
        rep = result.get("diagnostics", {}).get("repetition", {}).get("4_word_ngrams", {})
        if rep:
            fields.append("Repeated 4-gram positions: " + str(rep["repeated_positions"]))
        prior = [m["content"].strip() for m in turn["messages"] if m["role"] == "assistant"]
        fields.append("Repeats shared-context answer: " + ("yes" if prior and turn["answer"].strip() == prior[-1] else "no"))
        if previous_answer is not None:
            fields.append("Repeats own previous reply: " + ("yes" if turn["answer"].strip() == previous_answer.strip() else "no"))
    if side.get("error"):
        fields.append(str(side["error"]))
    return "<br>".join(html.escape(part) for part in fields)


def evidence(state, manifest, checksum, actor=None):
    item = state["rounds"][-1] if state.get("rounds") else None
    return {"current_candidate": state["candidate"], "history_policy": "Same history for both; candidate replies continue the shared conversation",
            "execution": "Sequential on one GPU: base then candidate; timings include loading/queueing, not a speed benchmark",
            "input_verified": bool(item and item.get("verified")),
            "input_messages": item["shared_messages"] if item else [],
            "messages_sha256": item.get("messages_sha256") if item else None,
            "base": demo.identity(manifest, checksum, "base"),
            "candidate": demo.identity(manifest, checksum, state["candidate"]),
            "generation": manifest["decoding"], "actor": actor or demo.actor_target(manifest),
            "checkpoint_directories": {name: spec["directory"] for name, spec in manifest["variants"].items() if spec},
            "native_prompt_hashes": {key: ((side.get("turn") or {}).get("remote_result") or {}).get("prompt_ids_sha256")
                                     for key, side in item["sides"].items()} if item else {},
            "assessment": "Generation checks do not score meaning, usefulness or clinical safety"}


def build_app(bundle, expected_sha, store=None, remote=None, cancel=demo.cancel_modal, *, app_name=None, class_name=None):
    manifest, module = demo.bound_bundle(bundle, expected_sha)
    store = demo.bundle_store(manifest, expected_sha, store)
    target = demo.actor_target(manifest, app_name, class_name)
    remote = remote if remote is not None else demo.routed_events(manifest, app_name, class_name)
    default = demo.default_variant(manifest)
    labels = {key: label for key, label in LABELS.items() if key in manifest["variants"]}
    r3 = default == "r3_response"
    if r3:
        labels["r2_meaning"] = labels["r2_meaning"].removesuffix(" (current)")
    import gradio as gr
    pair_store = comparison.comparison_store(store)
    active, lock = {}, threading.Lock()
    with gr.Blocks(title="Twi | Compare and chat", analytics_enabled=False) as app:
        gr.Markdown("# Twi " + ("R3" if r3 else "R2"))
        gr.Markdown("Messages are processed on private Modal compute. Chat history and reviews are saved on this device. Research, not medical advice.", elem_id="lab-notice")
        with gr.Tabs(selected="compare"):
            with gr.Tab("Compare", id="compare"):
                state = gr.State(None)
                with gr.Row():
                    history = gr.Dropdown(comparison.conversations(pair_store), label="Saved comparison", scale=4)
                    fresh = gr.Button("New comparison", size="sm", scale=1)
                with gr.Row(elem_id="compare-panes", equal_height=True):
                    with gr.Column(min_width=280):
                        gr.Markdown("### Base\nAfriqueQwen 9B | Adapter disabled\n\n`" + manifest["base_sha256"] + "`")
                        left = gr.Chatbot(show_label=False, container=False, layout="bubble", elem_classes="compare-chat",
                            height=390, buttons=["copy"], feedback_options=[], allow_file_downloads=False, sanitize_html=True)
                        left_metrics = gr.HTML("Awaiting comparison", elem_classes="evidence-strip")
                    with gr.Column(min_width=280):
                        right_label = gr.Markdown(candidate_heading(default, manifest))
                        right = gr.Chatbot(show_label=False, container=False, layout="bubble", elem_classes="compare-chat",
                            height=390, buttons=["copy"], feedback_options=[], allow_file_downloads=False, sanitize_html=True)
                        right_metrics = gr.HTML("Awaiting comparison", elem_classes="evidence-strip")
                status = gr.HTML("Ready", elem_id="comparison-status")
                composer = gr.Textbox(label="Shared prompt", placeholder="Twi or English", lines=2, max_lines=5, max_length=16000, submit_btn=True)
                stop_button = gr.Button("Stop", size="sm", variant="stop")
                with gr.Accordion("Your assessment", open=False):
                    round_choice = gr.Dropdown([], label="Compared turn")
                    rating = gr.Radio([("Base more useful", "base"), ("Candidate more useful", "candidate"),
                        ("Equal", "tie"), ("Neither", "neither"), ("Unsure", "unsure")], value="unsure", label="Usefulness")
                    note = gr.Textbox(label="Meaning, naturalness or correction", lines=2, max_length=8000)
                    save_review = gr.Button("Save assessment", size="sm")
                    review_status = gr.Markdown("")
                with gr.Accordion("Checkpoints and exact evidence", open=False):
                    choice = gr.Dropdown([(label, key) for key, label in labels.items()], value=default, label="Candidate checkpoint")
                    if not r3:
                        gr.Markdown("| R2 stage | Completed updates | Role |\n|---|---:|---|\n| Language | 1,081 | Twi text exposure |\n| Language + meaning | +873 | Bilingual alignment and intent; current selection |\n| Final response | +45 | Response targets; retained for comparison after regression |")
                    proof = gr.JSON({}, label="Input, model identities and generation settings", open=False)
            with gr.Tab("Chat", id="chat"):
                # Nested Blocks attach their children automatically; .render() would duplicate them.
                demo.build_app(bundle, expected_sha, store, remote, cancel, initialize=False, heading=False)

        def view(current, message=None):
            current = current or comparison.new_session(default)
            item = current["rounds"][-1] if current["rounds"] else None
            with lock:
                pending = current["id"] in active
            if message is None:
                if not item:
                    message = "Ready | Shared history: " + current["candidate"]
                elif item.get("verified"):
                    message = "Identical input and decoding verified | " + item["status"] + " | Quality: unreviewed"
                else:
                    message = item["status"] + " | Input comparison not yet verified"
                    if item.get("error"):
                        message += " | " + str(item["error"])
            choices = [(f"{i+1}: {r['question'][:65]}", r["id"]) for i, r in enumerate(current["rounds"])]
            saved = comparison.conversations(pair_store)
            previous = current["rounds"][-2] if len(current["rounds"]) > 1 else None
            def metrics(side):
                earlier = (previous["sides"][side].get("turn") or {}).get("answer") if previous else None
                return side_metrics(item["sides"][side] if item else None, earlier)
            return (current, transcript(current, "base"), transcript(current, "candidate"),
                metrics("base"), metrics("candidate"),
                html.escape(message), gr.update(value="", interactive=not pending),
                gr.update(value=current["candidate"], interactive=not pending), gr.update(interactive=not pending),
                gr.update(choices=saved, value=current["id"] if any(k == current["id"] for _, k in saved) else None, interactive=not pending),
                gr.update(choices=choices, value=choices[-1][1] if choices else None), evidence(current, manifest, expected_sha, target),
                candidate_heading(current["candidate"], manifest), "", gr.update(value="", interactive=not pending))

        outputs = [state, left, right, left_metrics, right_metrics, status, composer, choice, fresh, history,
                   round_choice, proof, right_label, review_status, note]

        async def respond(question, prior, candidate):
            current = prior or comparison.new_session(candidate)
            demo.require(current["candidate"] == candidate, "Checkpoint changed; start a new comparison")
            signal = threading.Event()
            last_error = None
            with lock:
                demo.require(current["id"] not in active, "Comparison already active")
                active[current["id"]] = signal
            try:
                async for current in comparison.compare(current, question, manifest, expected_sha, module, pair_store,
                        stop=signal, remote=remote, cancel=cancel):
                    yield view(current)
            except Exception as exc:
                last_error = str(exc)[:1500]
                yield view(current, last_error)
            finally:
                with lock:
                    active.pop(current["id"], None)
            yield view(current, last_error)

        options = {"api_visibility": "private", "concurrency_limit": 1, "concurrency_id": "r2-demo"}
        composer.submit(respond, [composer, state, choice], outputs, **options)
        def stop(prior):
            with lock:
                signal = active.get((prior or {}).get("id"))
                if signal:
                    signal.set()
        stop_button.click(stop, [state], None, queue=False, api_visibility="private")
        for event in (choice.input, fresh.click):
            event(lambda candidate: view(comparison.new_session(candidate)), [choice], outputs, **options)
        history.input(lambda key: view(comparison.load_session(pair_store, key)) if key else view(None), [history], outputs, **options)
        def first_view():
            saved = comparison.conversations(pair_store)
            return view(comparison.load_session(pair_store, saved[0][1]) if saved else None)
        app.load(first_view, outputs=outputs, api_visibility="private")
        def review(prior, selected, value, comment):
            try:
                comparison.save_review(pair_store, prior["id"], selected, value, comment)
                return "Assessment saved locally."
            except (ValueError, TypeError) as exc:
                raise gr.Error(str(exc)) from None
        save_review.click(review, [state, round_choice, rating, note], review_status, api_visibility="private")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--app-name")
    parser.add_argument("--class-name")
    parser.add_argument("--port", type=int, default=7866)
    args = parser.parse_args()
    manifest, _ = demo.bound_bundle(args.bundle, args.expected_sha)
    args.store = demo.bundle_store(manifest, args.expected_sha, args.store)
    demo.require(1024 <= args.port <= 65535, "Choose an unprivileged local port")
    import gradio as gr
    from starlette.middleware import Middleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    theme = gr.themes.Base(font=["system-ui", "sans-serif"], primary_hue="gray", secondary_hue="gray", neutral_hue="gray")
    theme.set(body_background_fill="#fafbfc", body_background_fill_dark="#fafbfc", body_text_color="#24262a",
        body_text_color_dark="#24262a", background_fill_primary="#ffffff", background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f0f2f4", background_fill_secondary_dark="#f0f2f4")
    build_app(args.bundle, args.expected_sha, args.store, app_name=args.app_name, class_name=args.class_name).queue(max_size=4).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False, show_error=False,
        strict_cors=True, enable_monitoring=False, mcp_server=False, ssr_mode=False, footer_links=[], theme=theme, css=CSS,
        blocked_paths=[str(args.bundle.resolve()), str(args.store.resolve().parent)],
        app_kwargs={"middleware": [Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])]})


if __name__ == "__main__":
    main()
