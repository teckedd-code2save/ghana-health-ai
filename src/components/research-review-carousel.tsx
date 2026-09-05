"use client";

import { useEffect, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Flag, Mic, Save, X } from "lucide-react";
import { recordUntilSilence } from "@/lib/browser-audio";

type Decision = "unreviewed" | "reviewed" | "needs_second_review" | "exclude";
type SafetyLevel = "" | "routine" | "same_day" | "urgent" | "emergency";
type CorpusFilter = "medical_large" | "language_sources" | "local_audio" | "product_text" | "all";

type AnnotationProposal = {
  proposal_id: string;
  agent_role: "source" | "translator" | "semantic_annotator" | "adjudicator";
  model: string;
  normalized_twi: string;
  natural_english: string;
  literal_english: string;
  intent: string;
  entities: string;
  ambiguities: string;
  reply_twi: string;
  safety_level: string;
  requires_clarification: boolean;
  score?: { total: number; flags: string[] };
};

type AnnotationSet = {
  prompt_version: string;
  proposals: AnnotationProposal[];
  recommended_proposal_id: string;
  synthesized_proposal: AnnotationProposal | null;
  adjudication: {
    confidence: number;
    status: "recommended" | "synthesized" | "needs_human_review";
    disagreement_fields: string[];
  };
};

type Review = {
  normalizedTwi: string;
  naturalEnglish: string;
  literalEnglish: string;
  intent: string;
  entities: string;
  ambiguities: string;
  replyTwi: string;
  safetyLevel: SafetyLevel;
  selectedProposalId: string;
  synthesisVersion: string;
  decision: Decision;
  notes: string;
};

type Row = {
  id: string;
  text: string;
  domain: string;
  language: string;
  source: string;
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
  annotationSet: AnnotationSet | null;
  review: (Review & { id: string }) | null;
};

type Payload = {
  candidates?: { rows?: Row[]; total?: number };
};

const PAGE_SIZE = 50;
const corpusOptions: Array<{ value: CorpusFilter; label: string; kind: string }> = [
  { value: "medical_large", label: "Medical corpus", kind: "Semantic corpus" },
  { value: "language_sources", label: "WAXAL + GhanaNLP", kind: "Language corpus" },
  { value: "local_audio", label: "Consented recordings", kind: "Voice corpus" },
  { value: "all", label: "All training drafts", kind: "Combined corpus" },
  { value: "product_text", label: "Response drafts", kind: "Direct response" },
];

function safetyLevel(value?: string): SafetyLevel {
  return value === "routine" || value === "same_day" || value === "urgent" || value === "emergency"
    ? value
    : "";
}

function proposalOptions(row: Row) {
  if (!row.annotationSet) return [];
  return [
    ...(row.annotationSet.synthesized_proposal ? [row.annotationSet.synthesized_proposal] : []),
    ...row.annotationSet.proposals,
  ];
}

function recommendedProposal(row: Row) {
  const options = proposalOptions(row);
  return options.find((proposal) => proposal.proposal_id === row.annotationSet?.recommended_proposal_id) ?? null;
}

function formFromProposal(row: Row, proposal: AnnotationProposal | null): Review {
  return {
    normalizedTwi: proposal?.normalized_twi || row.modelProposal.normalized_twi || row.text,
    naturalEnglish: proposal ? proposal.natural_english : row.modelProposal.natural_english,
    literalEnglish: proposal ? proposal.literal_english : row.modelProposal.literal_english,
    intent: proposal ? proposal.intent : row.modelProposal.intent,
    entities: proposal ? proposal.entities : row.modelProposal.entities,
    ambiguities: proposal ? proposal.ambiguities : row.modelProposal.ambiguities,
    replyTwi: proposal ? proposal.reply_twi : row.modelProposal.reply_twi,
    safetyLevel: safetyLevel(proposal ? proposal.safety_level : row.modelProposal.safety_level),
    selectedProposalId: proposal?.proposal_id ?? "",
    synthesisVersion: proposal ? row.annotationSet?.prompt_version ?? "" : "",
    decision: "unreviewed",
    notes: "",
  };
}

function formFor(row: Row): Review {
  return row.review ?? formFromProposal(row, recommendedProposal(row));
}

function proposalLabel(proposal: AnnotationProposal, recommendedId?: string) {
  const role = proposal.agent_role === "semantic_annotator"
    ? "Semantic agent"
    : proposal.agent_role === "translator"
      ? "Translation agent"
      : proposal.agent_role === "adjudicator"
        ? "Synthesis"
        : "Source annotation";
  return `${role}${proposal.proposal_id === recommendedId ? " · recommended" : ""}`;
}

function needsResponseReview(row: Row) {
  return Boolean(row.modelProposal.reply_twi.trim() || row.modelProposal.safety_level.trim());
}

export function ResearchReviewCarousel({ onClose }: { onClose: () => void }) {
  const [filter, setFilter] = useState<CorpusFilter>("medical_large");
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<Row[]>([]);
  const [total, setTotal] = useState(0);
  const [index, setIndex] = useState(0);
  const [form, setForm] = useState<Review | null>(null);
  const [speakerId, setSpeakerId] = useState("sp001");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordingCount, setRecordingCount] = useState(0);
  const [notice, setNotice] = useState("");

  const row = rows[index];
  const option = corpusOptions.find((item) => item.value === filter)!;

  useEffect(() => {
    let cancelled = false;
    void fetch(`/api/research/understanding?dataset=1&filter=${filter}&limit=${PAGE_SIZE}&offset=${offset}`, {
      cache: "no-store",
    })
      .then(async (res) => {
        const data = (await res.json()) as Payload & { error?: string };
        if (!res.ok) throw new Error(data.error || "Corpus rows are unavailable.");
        return data.candidates ?? {};
      })
      .then((candidates) => {
        if (cancelled) return;
        const nextRows = candidates.rows ?? [];
        setRows(nextRows);
        setTotal(candidates.total ?? 0);
        setIndex(0);
        setForm(nextRows[0] ? formFor(nextRows[0]) : null);
      })
      .catch((error: unknown) => {
        if (!cancelled) setNotice(error instanceof Error ? error.message : "Corpus rows are unavailable.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [filter, offset]);

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

  function selectCorpus(nextFilter: CorpusFilter) {
    setLoading(true);
    setNotice("");
    setFilter(nextFilter);
    setOffset(0);
  }

  function move(nextIndex: number) {
    if (!rows.length) return;
    const bounded = Math.max(0, Math.min(rows.length - 1, nextIndex));
    setIndex(bounded);
    setForm(formFor(rows[bounded]!));
    setNotice("");
  }

  function chooseProposal(proposalId: string) {
    if (!row) return;
    const proposal = proposalOptions(row).find((item) => item.proposal_id === proposalId);
    if (!proposal) return;
    const next = formFromProposal(row, proposal);
    setForm((current) => ({ ...next, decision: current?.decision ?? "unreviewed", notes: current?.notes ?? "" }));
    setNotice("");
  }

  function moveNext() {
    if (index < rows.length - 1) return move(index + 1);
    if (offset + rows.length < total) {
      setLoading(true);
      setOffset((current) => current + PAGE_SIZE);
    }
  }

  function movePrevious() {
    if (index > 0) return move(index - 1);
    if (offset > 0) {
      setLoading(true);
      setOffset((current) => Math.max(0, current - PAGE_SIZE));
    }
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
      if (decision === "reviewed") moveNext();
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

  if (loading) return <section className="research-review-card"><p>Loading corpus row…</p></section>;
  if (!row || !form) return <section className="research-review-card"><p>{notice || "No corpus rows available."}</p></section>;

  const responseReview = needsResponseReview(row);
  const ready = form.normalizedTwi.trim() && form.naturalEnglish.trim() && form.intent.trim() &&
    (!responseReview || (form.replyTwi.trim() && form.safetyLevel));
  const rowNumber = offset + index + 1;

  return (
    <section className="research-review-card" aria-label="Training corpus review">
      <header className="research-review-card__header">
        <div>
          <span>{option.kind} · Row {rowNumber.toLocaleString()} of {total.toLocaleString()}</span>
          <strong>Review training row</strong>
        </div>
        <button type="button" className="research-review-card__close" onClick={onClose} aria-label="Close review">
          <X className="h-4 w-4" />
        </button>
      </header>

      <label className="research-review-card__corpus">
        Corpus
        <select value={filter} onChange={(event) => selectCorpus(event.target.value as CorpusFilter)}>
          {corpusOptions.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </select>
      </label>
      <p className="research-review-card__source">{row.text}</p>

      {row.annotationSet && (
        <div className="research-review-card__proposal">
          <label>
            Annotation option
            <select
              value={form.selectedProposalId || recommendedProposal(row)?.proposal_id || ""}
              onChange={(event) => chooseProposal(event.target.value)}
            >
              {proposalOptions(row).map((proposal) => (
                <option key={proposal.proposal_id} value={proposal.proposal_id}>
                  {proposalLabel(proposal, row.annotationSet?.recommended_proposal_id)}
                </option>
              ))}
            </select>
          </label>
          <span>
            {Math.round(row.annotationSet.adjudication.confidence * 100)}% confidence
            {row.annotationSet.adjudication.disagreement_fields.length > 0
              ? ` · check ${row.annotationSet.adjudication.disagreement_fields.join(", ")}`
              : ""}
          </span>
        </div>
      )}

      <div className="research-review-card__fields">
        <label className="research-review-card__wide">
          Normalized Twi
          <textarea value={form.normalizedTwi} onChange={(event) => setForm({ ...form, normalizedTwi: event.target.value })} />
        </label>
        <label className="research-review-card__wide">
          English meaning
          <textarea value={form.naturalEnglish} onChange={(event) => setForm({ ...form, naturalEnglish: event.target.value })} />
        </label>
        <label>
          Intent
          <input value={form.intent} onChange={(event) => setForm({ ...form, intent: event.target.value })} />
        </label>
        {responseReview && <>
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
        </>}
      </div>

      <details className="research-review-card__details">
        <summary>More labels</summary>
        <div className="research-review-card__fields">
          <label>
            Entities
            <textarea value={form.entities} onChange={(event) => setForm({ ...form, entities: event.target.value })} />
          </label>
          <label>
            Ambiguity
            <textarea value={form.ambiguities} onChange={(event) => setForm({ ...form, ambiguities: event.target.value })} />
          </label>
        </div>
      </details>

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
          <button type="button" onClick={movePrevious} disabled={offset === 0 && index === 0} aria-label="Previous corpus row"><ChevronLeft className="h-4 w-4" /></button>
          <button type="button" onClick={() => void save()} disabled={saving} title="Save review"><Save className="h-4 w-4" /></button>
          <button type="button" onClick={() => void save("needs_second_review")} disabled={saving} title="Flag all options for another review" aria-label="Flag all options for another review"><Flag className="h-4 w-4" /></button>
          <button type="button" onClick={() => void save("reviewed")} disabled={saving || !ready} className="research-review-card__approve">
            <Check className="h-4 w-4" /> Review
          </button>
          <button type="button" onClick={moveNext} disabled={offset + index + 1 >= total} aria-label="Next corpus row"><ChevronRight className="h-4 w-4" /></button>
        </div>
      </footer>
      {notice && <p className="research-review-card__notice" role="status">{notice}</p>}
    </section>
  );
}
