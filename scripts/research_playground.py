"""Local, private chat with the actual saved adapter. No hosted LLM fallback.

Run: python3 scripts/research_playground.py --port 7863
"""
import argparse
import asyncio
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "tmp/research-playground/reviews.sqlite3"
APP_NAME = "ghana-understanding-pilot-private"
LABELS = {"afrique_v3": "Afrique Response v3", "afrique_base": "Afrique base (untrained for chat)",
          "afrique_v6": "Afrique Response v6 (research)", "afrique9_base": "Afrique 9B base (untrained for chat)",
          "response_v4": "Response v4 (research)", "response_v5": "Response v5 (research)",
          "response_v2": "Response v2", "response_base": "Gemma base", "pilot": "Pilot v1", "base": "Original base"}


def response_candidate(path=ROOT / "tmp/research-playground/private-selection.json"):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("enabled") is not True:
        return None
    for key, pattern in {"run_id": r"(?:response_v[245]|afrique_v[36])_\d{8}T\d{6}Z", "checkpoint": r"adapter|checkpoints/checkpoint-[1-9]\d{0,3}",
                         "adapter_sha256": r"[0-9a-f]{64}"}.items():
        if not isinstance(value.get(key), str) or not re.fullmatch(pattern, value[key]):
            raise ValueError("Private response model selection is invalid")
    return {key: value[key] for key in ("run_id", "checkpoint", "adapter_sha256")}


def candidate_variants(candidate):
    if candidate is None:
        return []
    if candidate["run_id"].startswith("afrique_v3_"):
        return ["afrique_v3", "afrique_base"]
    if candidate["run_id"].startswith("afrique_v6_"):
        return ["afrique_v6", "afrique9_base"]
    return ["response_" + candidate["run_id"].split("_")[1], "response_base"]


def browser_secret(path: Path = STORE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("INSERT OR IGNORE INTO settings VALUES ('browser_secret', ?)", (secrets.token_hex(24),))
        return db.execute("SELECT value FROM settings WHERE key = 'browser_secret'").fetchone()[0]


def save_turn(turn: dict, path: Path = STORE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS turns (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        db.execute("INSERT OR REPLACE INTO turns VALUES (?, ?)", (turn["id"], json.dumps(turn, ensure_ascii=False)))


def save_review(turn_id: str, rating: str, correction: str, path: Path = STORE):
    if rating not in ("Useful", "Incorrect", "Unsure") or len(correction) > 8000:
        raise ValueError("Invalid review")
    with sqlite3.connect(path) as db:
        if not db.execute("SELECT 1 FROM turns WHERE id = ?", (turn_id,)).fetchone():
            raise ValueError("Select a completed reply first")
        db.execute("CREATE TABLE IF NOT EXISTS reviews (turn_id TEXT PRIMARY KEY, rating TEXT, correction TEXT, updated REAL)")
        db.execute("INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?)", (turn_id, rating, correction, time.time()))


def existing_note(turn_id: str, path: Path = STORE):
    with sqlite3.connect(path) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name = 'reviews'").fetchone():
            return ""
        row = db.execute("SELECT correction FROM reviews WHERE turn_id = ?", (turn_id,)).fetchone()
        return row[0] if row else ""


def history_messages(turns: list[dict], question: str):
    if not isinstance(turns, list) or len(turns) >= 15:
        raise ValueError("Start a new conversation after 15 replies")
    result = []
    for turn in turns:
        if turn.get("status") == "complete":
            result.extend([{"role": "user", "content": turn["question"]},
                           {"role": "assistant", "content": turn["answer"]}])
    result.append({"role": "user", "content": question})
    return result


def display(turns):
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn["question"]})
        if turn.get("answer"):
            label = LABELS.get(turn.get("variant"), "Unknown model")
            suffix = " (incomplete)" if turn.get("status") in ("stopped", "failed") or turn.get("finish_reason") == "limit" else ""
            messages.append({"role": "assistant", "content": f"**{label}{suffix}**\n\n{turn['answer']}"})
    return messages


def restored(value):
    if not isinstance(value, list):
        return []
    clean = []
    for turn in value[-15:]:
        if (isinstance(turn, dict) and all(isinstance(turn.get(key), str) for key in ("id", "question", "answer"))
                and turn.get("variant") in LABELS):
            clean.append({**turn, "status": "stopped" if turn.get("status") == "streaming" else turn.get("status")})
    return clean


CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto; padding: 16px !important; }
.gradio-container .main { padding: 0 !important; }
body, .gradio-container { background: #fafbfc !important; color: #24262a !important; }
h1 { font-size: 24px !important; font-weight: 600 !important; letter-spacing: 0 !important; }
#conversation { border: 0 !important; background: transparent !important; }
#conversation .message { border-radius: 8px !important; box-shadow: none !important; }
#conversation .message p { line-height: 1.6; }
#conversation .message p:first-child strong { font-size: 12px; color: #656970; font-weight: 500; }
#conversation button[aria-label="Clear"] { display: none; }
#notice p, #status p { font-size: 13px; color: #697078; }
#status { min-height: 26px; }
#composer textarea { font-size: 16px; }
#voice-comparison { border: 0; border-top: 1px solid #dce0e5; border-radius: 0; background: transparent; }
#voice-comparison .form, #voice-comparison .block { border: 0 !important; box-shadow: none !important; background: transparent !important; }
#voice-save { min-height: 40px; }
.pending { display: inline-flex; gap: 5px; align-items: center; }
.pending i { width: 5px; height: 5px; background: #737982; border-radius: 50%; animation: dot 1.2s infinite; }
.pending i:nth-child(2) { animation-delay: .15s; } .pending i:nth-child(3) { animation-delay: .3s; }
@keyframes dot { 50% { opacity: .25; transform: translateY(-2px); } }
@media (prefers-reduced-motion: reduce) { .pending i { animation: none; } }
"""

# Gradio's embedded send/stop icons currently have no accessible names.
COMPOSER_LABELS = """() => {
  const label = () => {
    document.querySelectorAll('#composer button[data-testid]').forEach(button => {
      const name = button.dataset.testid === 'stop-button' ? 'Stop response' : 'Send message';
      button.setAttribute('aria-label', name);
      button.setAttribute('title', name);
    });
  };
  label();
  new MutationObserver(label).observe(document.body, {childList: true, subtree: true});
}"""


def build_app():
    import gradio as gr
    import modal

    candidate = response_candidate()
    direct_choices = candidate_variants(candidate)
    available = [*direct_choices, "pilot", "base"]
    with gr.Blocks(title="Understanding research", analytics_enabled=False) as demo:
        state = gr.State([])
        browser = gr.BrowserState([], storage_key="gha-private-pilot-conversation-v1", secret=browser_secret())
        selected_review = gr.State("")
        gr.Markdown("# Understanding research")
        gr.Markdown("Experimental model, not medical advice. Test text is processed on Modal; reviews stay on this device.", elem_id="notice")
        with gr.Row():
            variant = gr.Dropdown([(LABELS[key], key) for key in available], value=available[0], label="Model", scale=3)
            new = gr.Button("New conversation", size="sm", scale=1, min_width=140)
        chat = gr.Chatbot(show_label=False, container=False, height="clamp(220px, calc(100dvh - 370px), 650px)",
                          layout="bubble", buttons=["copy"], feedback_options=["Like", "Dislike"],
                          allow_file_downloads=False, elem_id="conversation")
        status = gr.HTML("", elem_id="status")
        composer = gr.Textbox(show_label=False, placeholder="Message in Twi or English", lines=2, max_lines=5,
                              max_length=8000, submit_btn=True, stop_btn=False, elem_id="composer")
        with gr.Accordion("Review a reply", open=False) as review:
            rating = gr.Radio(["Useful", "Incorrect", "Unsure"], label="Response", value="Incorrect")
            correction = gr.Textbox(label="Correction or note", lines=2, max_length=8000)
            save = gr.Button("Save review", size="sm")
            review_status = gr.Markdown("")

        async def respond(question, prior, choice):
            if not question.strip():
                return
            if choice not in available:
                raise gr.Error("That model is not enabled in this private tester")
            if len(prior) >= 15:
                raise gr.Error("Start a new conversation after 15 replies. Your existing replies remain saved locally.")
            # The prompt is only actual conversation, without UI labels or reference answers.
            messages = history_messages(prior, question.strip())
            turn = {"id": str(uuid.uuid4()), "question": question.strip(), "answer": "", "variant": choice,
                    "status": "streaming", "created": time.time(), "messages": messages}
            turns = [*prior, turn]
            call_id = None
            complete = False
            waiting = '<span class="pending" aria-label="Waiting for model"><i></i><i></i><i></i></span>'

            def output(note=""):
                return turns, turns, display(turns), note, gr.update(value="", submit_btn=False, stop_btn=True), gr.update(interactive=False), gr.update(interactive=False)

            yield output(waiting)
            try:
                engine = (modal.Cls.from_name("ghana-response-v2-private", "Response")(**candidate)
                          if choice in direct_choices else modal.Cls.from_name(APP_NAME, "Pilot")())
                async for event in engine.stream.remote_gen.aio(messages, choice):
                    if event["type"] == "started":
                        call_id = event["call_id"]
                        turn["provenance"] = event
                    elif event["type"] == "delta":
                        turn["answer"] += event["text"]
                    elif event["type"] in ("trace", "tool_result"):
                        turn.setdefault("inference_trace", []).append(event)
                        if event["type"] == "tool_result":
                            yield output("Calculating...")
                            continue
                    elif event["type"] == "done":
                        if not turn["answer"].strip():
                            raise ValueError("The model returned an empty reply")
                        complete = True
                        turn.update(status="complete", generation_seconds=event["seconds"],
                                    finish_reason=event.get("finish_reason"), output_tokens=event.get("output_tokens"))
                        save_turn(turn)
                    yield output(waiting if not turn["answer"] else "")
            except asyncio.CancelledError:
                turn["status"] = "stopped"
                raise
            except Exception as exc:
                turn["status"] = "failed"
                turn["error_type"] = type(exc).__name__
                yield output("Model request failed. Your conversation is kept; no fallback model was used.")
            finally:
                if turn["status"] == "streaming":
                    turn["status"] = "stopped"
                if not complete and call_id:
                    try:
                        await asyncio.wait_for(modal.FunctionCall.from_id(call_id).cancel.aio(), timeout=10)
                    except Exception:
                        turn["cancel_unconfirmed"] = True
                save_turn(turn)

        outputs = [state, browser, chat, status, composer, variant, new]
        event = composer.submit(respond, [composer, state, variant], outputs, concurrency_limit=1,
                                show_progress="hidden", api_visibility="private")

        def ready():
            return gr.update(submit_btn=True, stop_btn=False), gr.update(interactive=True), gr.update(interactive=True)

        event.then(ready, None, [composer, variant, new], queue=False)

        def stop(turns):
            turns = restored(turns)
            return turns, turns, display(turns), "Stopped", *ready()

        composer.stop(stop, state, outputs, cancels=[event], queue=False, api_visibility="private")

        def restore(value):
            turns = restored(value)
            return turns, display(turns)

        demo.load(restore, browser, [state, chat], queue=False)
        demo.load(fn=None, inputs=None, outputs=None, js=COMPOSER_LABELS, queue=False)
        new.click(lambda: ([], [], [], "", gr.update(open=False)), None,
                  [state, browser, chat, status, review], queue=False, api_visibility="private")

        def select_review(turns, data: gr.LikeData):
            index = data.index[0] if isinstance(data.index, (list, tuple)) else data.index
            rendered = display(turns)
            if index >= len(rendered) or rendered[index]["role"] != "assistant":
                raise gr.Error("Select a model reply")
            answer_index = sum(m["role"] == "assistant" for m in rendered[:index + 1]) - 1
            turn = [t for t in turns if t.get("answer")][answer_index]
            selected = "Useful" if data.liked else "Incorrect"
            note = existing_note(turn["id"])
            save_review(turn["id"], selected, note)
            return turn["id"], gr.update(visible=True, open=True), selected, note, "Rating saved locally."

        chat.like(select_review, state, [selected_review, review, rating, correction, review_status], api_visibility="private")

        def review_reply(turn_id, value, text):
            save_review(turn_id, value, text)
            return "Review saved locally."

        save.click(review_reply, [selected_review, rating, correction], review_status, api_visibility="private")

        from voice_listening import FOLDER, PHRASES, PRONUNCIATION, sample, save_voice_review, voice_choices
        if (FOLDER / "cosy-listening.json").exists() and (FOLDER / "listening.json").exists():
            loaded_voice = gr.State("")
            initial_choices = voice_choices("question")
            initial_voice = "cosy-twi-fp32-stream" if any(v == "cosy-twi-fp32-stream" for _, v in initial_choices) else "cosy-twi-owner-reference"
            with gr.Accordion("Twi voice comparison", open=False, elem_id="voice-comparison"):
                with gr.Row():
                    phrase = gr.Dropdown([(v, k) for k, v in PHRASES.items()], value="question", label="Phrase", min_width=220)
                    voice = gr.Dropdown(initial_choices, value=initial_voice, label="Voice", min_width=220)
                sample_text = gr.Textbox(label="Spoken text", interactive=False, lines=2)
                audition = gr.Audio(label="Voice sample", interactive=False, autoplay=False, buttons=[])
                pronunciation = gr.Radio(list(PRONUNCIATION), label="Pronunciation", value=None)
                naturalness = gr.Radio([1, 2, 3, 4, 5], label="Naturalness (1 poor, 5 natural)", value=None)
                voice_note = gr.Textbox(label="Words to correct or note", lines=2, max_length=2000)
                voice_save = gr.Button("Save voice review", size="sm", elem_id="voice-save")
                voice_status = gr.Markdown("")

                def audition_sample(phrase_id, voice_id):
                    choices = voice_choices(phrase_id)
                    if voice_id not in {value for _, value in choices}:
                        voice_id = "cosy-twi-owner-reference"
                    row, wav = sample(phrase_id, voice_id)
                    return gr.update(choices=choices, value=voice_id), row["text"], wav, None, None, "", "", f"{phrase_id}:{voice_id}"

                audition_outputs = [voice, sample_text, audition, pronunciation, naturalness, voice_note, voice_status, loaded_voice]
                gr.on([phrase.input, voice.input, demo.load], audition_sample, [phrase, voice],
                      audition_outputs, api_visibility="private", show_progress="hidden", trigger_mode="always_last")

                def record_voice_review(phrase_id, voice_id, words, natural, note, loaded):
                    if loaded != f"{phrase_id}:{voice_id}":
                        raise gr.Error("Wait for the selected voice sample to load")
                    try:
                        save_voice_review(phrase_id, voice_id, words, natural, note, path=STORE)
                    except ValueError as exc:
                        raise gr.Error(str(exc)) from None
                    return "Voice review saved locally."

                voice_save.click(record_voice_review, [phrase, voice, pronunciation, naturalness, voice_note, loaded_voice],
                                 voice_status, api_visibility="private")
    return demo


def main():
    import gradio as gr
    from starlette.middleware import Middleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7863)
    args = parser.parse_args()
    theme = gr.themes.Base(font=["system-ui", "sans-serif"], primary_hue="gray", secondary_hue="gray", neutral_hue="gray")
    theme.set(body_background_fill="#fafbfc", body_background_fill_dark="#fafbfc",
              body_text_color="#24262a", body_text_color_dark="#24262a",
              background_fill_primary="#ffffff", background_fill_primary_dark="#ffffff",
              background_fill_secondary="#f0f2f4", background_fill_secondary_dark="#f0f2f4")
    build_app().queue(max_size=4).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, show_error=False,
        strict_cors=True, enable_monitoring=False, mcp_server=False, ssr_mode=False,
        footer_links=[], theme=theme, css=CSS, blocked_paths=[str(ROOT / "tmp")],
        app_kwargs={"middleware": [Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])]},
    )


if __name__ == "__main__":
    main()
