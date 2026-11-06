"use client";

import { useRef, useState } from "react";
import { CircleHelp, LoaderCircle, MessageSquare, Save, ThumbsDown, ThumbsUp } from "lucide-react";

type Assessment = "useful" | "incorrect" | "unsure";

const assessments = [
  { value: "useful", label: "Useful", Icon: ThumbsUp },
  { value: "incorrect", label: "Incorrect", Icon: ThumbsDown },
  { value: "unsure", label: "Unsure", Icon: CircleHelp },
] as const;

export function ResearchReplyFeedback({ conversationId, messageId }: { conversationId: string; messageId: string }) {
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);

  async function save() {
    if (!assessment || !conversationId || !messageId || inFlight.current) return;
    inFlight.current = true;
    setSaving(true);
    setSaved(false);
    setError(null);
    try {
      const response = await fetch("/api/research/reply-feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversationId, messageId, assessment, note }),
      });
      if (!response.ok) {
        throw new Error(response.status === 401 || response.status === 403
          ? "Sign in to save research feedback."
          : "Feedback was not saved. Please try again.");
      }
      setSaved(true);
    } catch (error) {
      setError(error instanceof Error && error.message === "Sign in to save research feedback."
        ? error.message : "Feedback was not saved. Please try again.");
    } finally {
      inFlight.current = false;
      setSaving(false);
    }
  }

  return (
    <details className="mt-2 max-w-sm text-xs">
      <summary
        className="inline-flex h-8 w-8 cursor-pointer list-none items-center justify-center rounded hover:bg-black/5 focus-visible:outline-2 focus-visible:outline-offset-2 [&::-webkit-details-marker]:hidden"
        aria-label="Review research reply"
        title="Review research reply"
      >
        <MessageSquare className="h-4 w-4" aria-hidden="true" />
      </summary>
      <form className="mt-2 space-y-2" onSubmit={(event) => { event.preventDefault(); void save(); }}>
        <div className="flex gap-2" role="group" aria-label="Reply assessment">
          {assessments.map(({ value, label, Icon }) => (
            <button
              key={value}
              type="button"
              aria-label={label}
              title={label}
              aria-pressed={assessment === value}
              disabled={saving}
              className={`inline-flex h-9 w-9 items-center justify-center rounded border focus-visible:outline-2 focus-visible:outline-offset-2 disabled:opacity-50 ${assessment === value ? "border-current bg-black/10" : "border-current/15 hover:bg-black/5"}`}
              onClick={() => { setAssessment(value); setSaved(false); setError(null); }}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
            </button>
          ))}
        </div>
        <textarea
          value={note}
          maxLength={4000}
          rows={2}
          aria-label="Feedback note (optional)"
          placeholder="Optional note"
          disabled={saving}
          className="block w-full resize-y rounded border border-current/20 bg-transparent px-2 py-1.5 text-sm focus-visible:outline-2 focus-visible:outline-offset-2"
          onChange={(event) => { setNote(event.target.value); setSaved(false); setError(null); }}
        />
        <button
          type="submit"
          disabled={!assessment || saving || saved}
          className="inline-flex items-center gap-1.5 rounded px-2 py-1.5 hover:bg-black/5 focus-visible:outline-2 focus-visible:outline-offset-2 disabled:opacity-50"
        >
          {saving ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <Save className="h-3.5 w-3.5" aria-hidden="true" />}
          {saving ? "Saving..." : "Save feedback"}
        </button>
        {saved && <p role="status">Feedback saved.</p>}
        {error && <p role="alert">{error}</p>}
      </form>
    </details>
  );
}
