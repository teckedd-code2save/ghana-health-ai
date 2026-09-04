"use client";

import { useEffect, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Mic, Save, X } from "lucide-react";
import { recordUntilSilence } from "@/lib/browser-audio";

type Decision = "unreviewed" | "reviewed" | "needs_second_review" | "exclude";
type SafetyLevel = "" | "routine" | "same_day" | "urgent" | "emergency";

type Review = {
  normalizedTwi: string;
  naturalEnglish: string;
  literalEnglish: string;
  intent: string;
  entities: string;
  ambiguities: string;
  replyTwi: string;
  safetyLevel: SafetyLevel;
  decision: Decision;
  notes: string;
};

type Row = {
  id: string;
  text: string;
  domain: string;
  language: string;
  modelProposal: {
    normalized_twi: string;
    natural_english: string;
    literal_english: string;
    intent: string;
    entities: string;
    ambiguities: string;
    reply_twi: string;
    safety_level: string;
  };
  review: (Review & { id: string }) | null;
};

type Payload = { candidates?: { rows?: Row[] } };

function safetyLevel(value?: string): SafetyLevel {
  return value === "routine" || value === "same_day" || value === "urgent" || value === "emergency"
    ? value
    : "";
}

function formFor(row: Row): Review {
  return row.review ?? {
    normalizedTwi: row.modelProposal.normalized_twi || row.text,
    naturalEnglish: row.modelProposal.natural_english,
    literalEnglish: row.modelProposal.literal_english,
    intent: row.modelProposal.intent,
    entities: row.modelProposal.entities,
    ambiguities: row.modelProposal.ambiguities,
    replyTwi: row.modelProposal.reply_twi,
    safetyLevel: safetyLevel(row.modelProposal.safety_level),
    decision: "unreviewed",
    notes: "",
  };
}

export function ResearchReviewCarousel({ onClose }: { onClose: () => void }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [index, setIndex] = useState(0);
  const [form, setForm] = useState<Review | null>(null);
  const [speakerId, setSpeakerId] = useState("sp001");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordingCount, setRecordingCount] = useState(0);
  const [notice, setNotice] = useState("");

  const row = rows[index];

  useEffect(() => {
    let cancelled = false;
    void fetch("/api/research/understanding?dataset=1&filter=product_text&limit=24", {
      cache: "no-store",
    })
      .then(async (res) => {
        const data = (await res.json()) as Payload & { error?: string };
        if (!res.ok) throw new Error(data.error || "Review samples are unavailable.");
        return data.candidates?.rows ?? [];
      })
      .then((nextRows) => {
        if (cancelled) return;
        setRows(nextRows);
        setIndex(0);
        setForm(nextRows[0] ? formFor(nextRows[0]) : null);
      })
      .catch((error: unknown) => {
        if (!cancelled) setNotice(error instanceof Error ? error.message : "Review samples are unavailable.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const rowId = row?.id;

  useEffect(() => {
    if (!rowId) return;
    void fetch(`/api/research/understanding/recordings?rowId=${encodeURIComponent(rowId)}`, {
      cache: "no-store",
    })
      .then((res) => res.ok ? res.json() : { recordings: [] })
      .then((data: { recordings?: unknown[] }) => setRecordingCount(data.recordings?.length ?? 0))
      .catch(() => setRecordingCount(0));
  }, [rowId]);

  function move(nextIndex: number) {
    if (!rows.length) return;
    const bounded = Math.max(0, Math.min(rows.length - 1, nextIndex));
    setIndex(bounded);
    setForm(formFor(rows[bounded]!));
    setNotice("");
  }

  async function save(decision?: Decision) {
    if (!row || !form) return;
    setSaving(true);
    setNotice("");
    const review = { ...form, id: row.id, ...(decision ? { decision } : {}) };
    try {
      const res = await fetch("/api/research/understanding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(review),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Review could not be saved.");
      setForm((current) => current ? { ...current, decision: review.decision } : current);
      setNotice(review.decision === "reviewed" ? "Reviewed" : "Saved");
      if (decision === "reviewed") move(index + 1);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Review could not be saved.");
    } finally {
      setSaving(false);
    }
  }

  async function record() {
    if (!row || !form || recording) return;
    setRecording(true);
    setNotice("");
    try {
      const result = await recordUntilSilence({ silenceMs: 1150, maxMs: 30_000, minSpeechMs: 500 });
      if (!result.speechDetected || result.blob.size < 400) {
        setNotice("No speech captured.");
        return;
      }
      const body = new FormData();
      body.set("audio", result.blob, `${row.id}-${speakerId}.webm`);
      body.set("rowId", row.id);
      body.set("speakerId", speakerId.trim() || "sp001");
      body.set("transcript", form.normalizedTwi || row.text);
      body.set("language", row.language === "ga" ? "ga" : row.language === "en" ? "en" : "tw");
      body.set("durationMs", String(Math.round(result.durationMs)));
      body.set("consent", "true");
      const res = await fetch("/api/research/understanding/recordings", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Recording could not be saved.");
      setRecordingCount((count) => count + 1);
      setNotice("Voice sample saved");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Recording failed.");
    } finally {
      setRecording(false);
    }
  }

  if (loading) return <section className="research-review-card"><p>Loading review sample…</p></section>;
  if (!row || !form) return <section className="research-review-card"><p>{notice || "No review samples available."}</p></section>;

  const ready = form.normalizedTwi.trim() && form.naturalEnglish.trim() && form.intent.trim() &&
    (row.domain !== "health" || (form.replyTwi.trim() && form.safetyLevel));

  return (
    <section className="research-review-card" aria-label="Research sample review">
      <header className="research-review-card__header">
        <div>
          <span>Research sample {index + 1} of {rows.length}</span>
          <strong>{row.domain === "health" ? "Health response review" : "Language review"}</strong>
        </div>
        <button type="button" className="research-review-card__close" onClick={onClose} aria-label="Close review">
          <X className="h-4 w-4" />
        </button>
      </header>

      <p className="research-review-card__source">{row.text}</p>

      <div className="research-review-card__fields">
        <label>
          Meaning
          <textarea value={form.naturalEnglish} onChange={(event) => setForm({ ...form, naturalEnglish: event.target.value })} />
        </label>
        <label>
          Intent
          <input value={form.intent} onChange={(event) => setForm({ ...form, intent: event.target.value })} />
        </label>
        <label className="research-review-card__wide">
          Twi response
          <textarea value={form.replyTwi} onChange={(event) => setForm({ ...form, replyTwi: event.target.value })} />
        </label>
        <label>
          Safety
          <select value={form.safetyLevel} onChange={(event) => setForm({ ...form, safetyLevel: safetyLevel(event.target.value) })}>
            <option value="">Choose</option>
            <option value="routine">Routine</option>
            <option value="same_day">Same day</option>
            <option value="urgent">Urgent</option>
            <option value="emergency">Emergency</option>
          </select>
        </label>
      </div>

      <footer className="research-review-card__footer">
        <div className="research-review-card__record">
          <input value={speakerId} onChange={(event) => setSpeakerId(event.target.value)} aria-label="Speaker ID" />
          <button type="button" onClick={() => void record()} disabled={recording} title="Record this phrase with consent">
            <Mic className="h-4 w-4" />
            {recording ? "Listening" : "Record"}
          </button>
          {recordingCount > 0 && <span>{recordingCount} clip{recordingCount === 1 ? "" : "s"}</span>}
        </div>
        <div className="research-review-card__actions">
          <button type="button" onClick={() => move(index - 1)} disabled={index === 0} aria-label="Previous sample"><ChevronLeft className="h-4 w-4" /></button>
          <button type="button" onClick={() => void save()} disabled={saving} title="Save review"><Save className="h-4 w-4" /></button>
          <button type="button" onClick={() => void save("reviewed")} disabled={saving || !ready} className="research-review-card__approve">
            <Check className="h-4 w-4" /> Review
          </button>
          <button type="button" onClick={() => move(index + 1)} disabled={index === rows.length - 1} aria-label="Next sample"><ChevronRight className="h-4 w-4" /></button>
        </div>
      </footer>
      {notice && <p className="research-review-card__notice" role="status">{notice}</p>}
    </section>
  );
}
