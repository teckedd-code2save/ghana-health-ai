"""Private, checksum-verified auditions; no source-recording file serving."""
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "tmp/twi-voice-comparison"
VOICES = {"cosy-twi-owner-reference": "CosyVoice",
          "cosy-twi-streaming": "CosyVoice stream",
          "cosy-twi-fp32-stream": "CosyVoice A",
          "cosy-twi-fp16-whole": "CosyVoice B",
          "stable-twi-6": "Stable Twi 6", "stable-twi-1": "Stable Twi 1", "nano-4step": "Nano Twi"}
TAIL_VARIANTS = {"cosy-twi-fp32-stream": "fp32-stream", "cosy-twi-fp16-whole": "fp16-whole"}
PHRASES = {"greeting": "Greeting", "question": "Feeling unwell", "shopping": "Shopping"}
PRONUNCIATION = ("Correct words", "Mispronounced or missing words", "Unsure")


def voice_choices(phrase, folder=FOLDER):
    if phrase not in PHRASES:
        raise ValueError("Unknown voice phrase")
    tail_available = phrase == "question" and (folder / "cosy-tail-listening.json").exists()
    return [(label, voice) for voice, label in VOICES.items()
            if voice not in TAIL_VARIANTS or tail_available]


def sample(phrase, voice, folder=FOLDER):
    if phrase not in PHRASES or voice not in VOICES:
        raise ValueError("Unknown voice sample")
    manifest = "cosy-listening.json" if voice == "cosy-twi-owner-reference" else "listening.json"
    if voice == "cosy-twi-streaming":
        manifest = "cosy-streaming-listening.json"
    if voice in TAIL_VARIANTS:
        manifest = "cosy-tail-listening.json"
    rows = json.loads((folder / manifest).read_text())["results"]
    if voice in TAIL_VARIANTS:
        matches = [r for r in rows if r["id"] == phrase and r.get("variant") == TAIL_VARIANTS[voice]
                   and r["voice"] == "cosy-twi-owner-reference"]
    else:
        matches = [r for r in rows if r["id"] == phrase and r["voice"] == voice]
    if len(matches) > 1:
        raise ValueError("Ambiguous voice sample")
    row = matches[0] if matches else None
    if row is None:
        raise ValueError("Voice sample is unavailable")
    filename = f"listen-{phrase}-{voice}.wav"
    if voice == "cosy-twi-streaming":
        filename = f"listen-stream-fp16-{phrase}-cosy-twi-owner-reference.wav"
    if voice in TAIL_VARIANTS:
        filename = f"listen-tail-{TAIL_VARIANTS[voice]}-{phrase}.wav"
        row = {**row, "synthesis_voice": row["voice"], "voice": voice}
    if row.get("listening_file") != filename:
        raise ValueError("Unexpected listening file")
    path = folder / filename
    if path.is_symlink() or path.resolve().parent != folder.resolve():
        raise ValueError("Unexpected listening file location")
    audio = path.read_bytes()
    if hashlib.sha256(audio).hexdigest() != row.get("listening_sha256"):
        raise ValueError("Voice sample integrity check failed")
    return row, audio


def save_tail_preference(choice, *, path, folder=FOLDER):
    """Record the owner's A/B/original choice, not an absolute quality score."""
    options = {"A": "cosy-twi-fp32-stream", "B": "cosy-twi-fp16-whole", "original": "cosy-twi-streaming"}
    if choice not in options:
        raise ValueError("Unknown tail-comparison choice")
    samples = {label: sample("question", voice, folder)[0] for label, voice in options.items()}
    if len({row["text"] for row in samples.values()}) != 1:
        raise ValueError("Preference samples must contain the same sentence")
    identity = json.dumps({"choice": choice, "samples": {label: row["listening_sha256"]
                           for label, row in samples.items()}}, sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()
    record = {"comparison": "cosy-owner-reference-question-tail", "owner_response": choice,
              "selected": samples[choice], "options": samples, "source": "owner_chat_feedback",
              "question": "Which has the clearer ending while still sounding like you: A, B, or the original?",
              "pronunciation": None, "naturalness": None, "training_eligible": False,
              "scope": "Preference for this sentence only; no production promotion", "updated": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS voice_preferences (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO voice_preferences VALUES (?, ?)", (key, json.dumps(record, ensure_ascii=False)))
        return json.loads(db.execute("SELECT record FROM voice_preferences WHERE id = ?", (key,)).fetchone()[0])


def save_voice_review(phrase, voice, pronunciation, naturalness, note, *, path, folder=FOLDER):
    if pronunciation not in PRONUNCIATION or naturalness not in (1, 2, 3, 4, 5) or not isinstance(note, str) or len(note) > 2000:
        raise ValueError("Choose pronunciation and naturalness ratings before saving")
    row, _ = sample(phrase, voice, folder)
    record = {"sample": row, "pronunciation": pronunciation, "naturalness": naturalness,
              "note": note.strip(), "updated": time.time(), "training_eligible": False}
    key = f"{phrase}:{voice}:{row['listening_sha256']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS voice_reviews (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        db.execute("INSERT OR REPLACE INTO voice_reviews VALUES (?, ?)", (key, json.dumps(record, ensure_ascii=False)))
    return record


def save_voice_observation(phrase, voice, note, *, path, folder=FOLDER):
    """Keep free-form owner feedback without inventing missing rating scores."""
    if not isinstance(note, str) or not note.strip() or len(note) > 2000:
        raise ValueError("Expected a nonempty owner observation")
    row, _ = sample(phrase, voice, folder)
    identity = f"{phrase}:{voice}:{row['listening_sha256']}:{note.strip()}"
    key = hashlib.sha256(identity.encode()).hexdigest()
    record = {"sample": row, "note": note.strip(), "source": "owner_chat_feedback",
              "pronunciation": None, "naturalness": None,
              "updated": time.time(), "training_eligible": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS voice_observations (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO voice_observations VALUES (?, ?)", (key, json.dumps(record, ensure_ascii=False)))
        return json.loads(db.execute("SELECT record FROM voice_observations WHERE id = ?", (key,)).fetchone()[0])
