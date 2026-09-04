"use client";

import { Dispatch, SetStateAction, useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Check, FileDown, Save, Upload } from "lucide-react";

type ReviewDecision = "unreviewed" | "reviewed" | "needs_second_review" | "exclude";
type CorpusFilter = "medical_large" | "language_sources" | "local_audio" | "product_text" | "all";

type Review = {
  id: string;
  normalizedTwi: string;
  naturalEnglish: string;
  literalEnglish: string;
  intent: string;
  entities: string;
  ambiguities: string;
  decision: ReviewDecision;
  notes: string;
};

type DatasetRow = {
  kind: "benchmark" | "corpus";
  id: string;
  category: string;
  text: string;
  review_status: string;
  source?: string;
  sourceRecordId?: string;
  split?: string;
  trainingSplit?: "train" | "dev" | "test";
  language?: string;
  speakerId?: string | null;
  audioArtifactId?: string | null;
  consentScope?: string;
  modelProposal?: {
    normalized_twi: string;
    natural_english: string;
    literal_english: string;
    intent: string;
    entities: string;
    ambiguities: string;
    requires_clarification: boolean;
    model: string;
    status: "not_requested" | "draft";
  };
  review: Review | null;
};

type WorkbenchPayload = {
  reviewer: string;
  candidates: {
    rows: DatasetRow[];
    total: number;
    visible: number;
    offset: number;
    limit: number;
    filter: CorpusFilter;
    completed: number;
    draftAnnotated: number;
    trainingReady: number;
    sourceSummary: Record<
      string,
      { total: number; draftAnnotated: number; reviewed: number; excluded: number }
    >;
    readiness: {
      ready: boolean;
      required_passed: number;
      required_total: number;
    };
  };
};

const PAGE_SIZE = 150;

const corpusFilterOptions: Array<[CorpusFilter, string]> = [
  ["medical_large", "Medical corpus"],
  ["language_sources", "WAXAL / GhanaNLP"],
  ["local_audio", "Local audio"],
  ["product_text", "Product rows"],
  ["all", "All rows"],
];

function emptyReview(row?: DatasetRow): Review {
  return {
    id: row?.id ?? "",
    normalizedTwi: row?.modelProposal?.normalized_twi || row?.text || "",
    naturalEnglish: row?.modelProposal?.natural_english ?? "",
    literalEnglish: row?.modelProposal?.literal_english ?? "",
    intent: row?.modelProposal?.intent ?? "",
    entities: row?.modelProposal?.entities ?? "",
    ambiguities: row?.modelProposal?.ambiguities ?? "",
    decision: row?.review?.decision ?? "unreviewed",
    notes: row?.review?.notes ?? "",
  };
}

function hydrateReview(row: DatasetRow): Review {
  return row.review ?? emptyReview(row);
}

function rowStatus(row: DatasetRow) {
  if (row.review?.decision === "reviewed") return "Reviewed";
  if (row.review?.decision === "needs_second_review") return "Second review";
  if (row.review?.decision === "exclude") return "Excluded";
  if (row.modelProposal?.status === "draft") return "Needs review";
  return "Missing annotation";
}

function sourceLabel(row: DatasetRow) {
  return [row.source ?? row.kind, row.trainingSplit ?? row.split, row.language]
    .filter(Boolean)
    .join(" / ");
}

export function UnderstandingWorkbench() {
  const [payload, setPayload] = useState<WorkbenchPayload | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [filter, setFilter] = useState<CorpusFilter>("medical_large");
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");

  async function load(nextFilter = filter, nextOffset = offset) {
    const query = new URLSearchParams({
      dataset: "1",
      filter: nextFilter,
      offset: String(nextOffset),
      limit: String(PAGE_SIZE),
    });
    const res = await fetch(`/api/research/understanding?${query}`, { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) {
      setError(data.error ?? "Dataset review is unavailable.");
      return;
    }
    setError("");
    setPayload(data);
  }

  useEffect(() => {
    let cancelled = false;
    async function run() {
      const query = new URLSearchParams({
        dataset: "1",
        filter,
        offset: String(offset),
        limit: String(PAGE_SIZE),
      });
      const res = await fetch(`/api/research/understanding?${query}`, { cache: "no-store" });
      const data = await res.json();
      if (cancelled) return;
      if (!res.ok) {
        setError(data.error ?? "Dataset review is unavailable.");
        return;
      }
      setError("");
      setPayload(data);
    }
    void run();
    return () => {
      cancelled = true;
    };
  }, [filter, offset]);

  const rows = useMemo(() => payload?.candidates.rows ?? [], [payload]);
  const selected = rows[selectedIndex] ?? rows[0];
  const pageStart = payload && rows.length ? payload.candidates.offset + 1 : 0;
  const pageEnd = payload
    ? Math.min(payload.candidates.offset + rows.length, payload.candidates.total)
    : 0;
  const reviewedOnPage = useMemo(
    () => rows.filter((row) => row.review?.decision === "reviewed").length,
    [rows],
  );
  const hasPrevPage = offset > 0;
  const hasNextPage = payload ? offset + PAGE_SIZE < payload.candidates.total : false;

  async function save(review: Review, nextIndex?: number) {
    if (!selected) return;
    setSaving(true);
    setError("");
    const res = await fetch("/api/research/understanding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...review, id: selected.id }),
    });
    const data = await res.json();
    setSaving(false);
    if (!res.ok) {
      setError(data.error ?? "Review could not be saved.");
      return;
    }
    await load();
    if (typeof nextIndex === "number") setSelectedIndex(Math.max(0, Math.min(rows.length - 1, nextIndex)));
  }

  async function uploadReviewSheet(file: File | null) {
    if (!file) return;
    setUploading(true);
    setUploadStatus("");
    setError("");
    const form = new FormData();
    form.set("file", file);
    const res = await fetch("/api/research/understanding/review-sheet", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    setUploading(false);
    if (!res.ok) {
      setError(data.error ?? "Review sheet could not be imported.");
      return;
    }
    setUploadStatus(`Imported ${data.imported} rows, skipped ${data.skipped}.`);
    await load();
  }

  return (
    <section className="research-ase">
      <header className="research-ase__header">
        <div>
          <span>Dataset Review</span>
          <h1>Understanding corpus</h1>
        </div>
        {payload && (
          <dl className="research-ase__stats">
            <div>
              <dt>Total</dt>
              <dd>{payload.candidates.total.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Reviewed</dt>
              <dd>{payload.candidates.completed.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Ready</dt>
              <dd>{payload.candidates.trainingReady.toLocaleString()}</dd>
            </div>
          </dl>
        )}
      </header>

      <div className="research-ase__toolbar">
        <label>
          Source
          <select
            value={filter}
            onChange={(event) => {
              setFilter(event.target.value as CorpusFilter);
              setOffset(0);
              setSelectedIndex(0);
            }}
          >
            {corpusFilterOptions.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <div className="research-ase__toolbar-actions">
          <a href="/api/research/understanding/export" target="_blank" rel="noreferrer">
            <FileDown className="h-4 w-4" />
            Export
          </a>
          <label className="research-ase__upload">
            <Upload className="h-4 w-4" />
            {uploading ? "Importing" : "Import CSV"}
            <input
              type="file"
              accept=".csv,text/csv"
              disabled={uploading}
              onChange={(event) => {
                const file = event.currentTarget.files?.[0] ?? null;
                void uploadReviewSheet(file);
                event.currentTarget.value = "";
              }}
            />
          </label>
        </div>
      </div>

      {error && <p className="research-ase__error">{error}</p>}
      {uploadStatus && <p className="research-ase__success">{uploadStatus}</p>}

      {payload && (
        <div className="research-ase__pagebar">
          <span>
            Showing {pageStart.toLocaleString()}-{pageEnd.toLocaleString()} of{" "}
            {payload.candidates.total.toLocaleString()}
          </span>
          <span>{reviewedOnPage.toLocaleString()} reviewed on this page</span>
          <div>
            <button
              type="button"
              disabled={!hasPrevPage}
              onClick={() => {
                setOffset(Math.max(0, offset - PAGE_SIZE));
                setSelectedIndex(0);
              }}
            >
              <ArrowLeft className="h-4 w-4" />
              Previous
            </button>
            <button
              type="button"
              disabled={!hasNextPage}
              onClick={() => {
                setOffset(offset + PAGE_SIZE);
                setSelectedIndex(0);
              }}
            >
              Next
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {selected ? (
        <div className="research-ase__review">
          <aside className="research-ase__queue" aria-label="Dataset queue">
            {rows.map((row, index) => (
              <button
                key={row.id}
                className={index === selectedIndex ? "is-active" : ""}
                onClick={() => setSelectedIndex(index)}
                type="button"
              >
                <strong>{row.text || row.id}</strong>
                <small>
                  {rowStatus(row)} · {sourceLabel(row)}
                </small>
              </button>
            ))}
          </aside>

          <ReviewEditor
            key={selected.id}
            row={selected}
            rowNumber={offset + selectedIndex + 1}
            rowCount={payload?.candidates.total ?? rows.length}
            pageIndex={selectedIndex}
            pageLength={rows.length}
            saving={saving}
            setSelectedIndex={setSelectedIndex}
            save={save}
          />
        </div>
      ) : (
        <p className="research-ase__empty">No rows found for this source.</p>
      )}
    </section>
  );
}

function ReviewEditor({
  row,
  rowNumber,
  rowCount,
  pageIndex,
  pageLength,
  saving,
  setSelectedIndex,
  save,
}: {
  row: DatasetRow;
  rowNumber: number;
  rowCount: number;
  pageIndex: number;
  pageLength: number;
  saving: boolean;
  setSelectedIndex: Dispatch<SetStateAction<number>>;
  save: (review: Review, nextIndex?: number) => Promise<void>;
}) {
  const [form, setForm] = useState<Review>(hydrateReview(row));
  const canAccept =
    form.normalizedTwi.trim().length > 0 &&
    form.naturalEnglish.trim().length > 0 &&
    form.intent.trim().length > 0;
  const nextIndex = Math.min(pageLength - 1, pageIndex + 1);

  return (
    <main className="research-ase__editor">
      <div className="research-ase__prompt">
        <div>
          <span>
            Row {rowNumber.toLocaleString()} / {rowCount.toLocaleString()}
          </span>
          <strong>{rowStatus(row)}</strong>
        </div>
        <p>{row.text}</p>
        <small>
          {sourceLabel(row)} · {row.consentScope ?? "consent unknown"} ·{" "}
          {row.speakerId ? `speaker ${row.speakerId}` : "speaker unknown"}
        </small>
        {row.audioArtifactId && <small>{row.audioArtifactId}</small>}
      </div>

      <div className="research-ase__form-grid">
        <label className="research-ase__field research-ase__field--wide">
          Normalized Twi
          <textarea
            value={form.normalizedTwi}
            onChange={(event) => setForm((value) => ({ ...value, normalizedTwi: event.target.value }))}
          />
        </label>
        <label className="research-ase__field research-ase__field--wide">
          Faithful English meaning
          <textarea
            value={form.naturalEnglish}
            onChange={(event) => setForm((value) => ({ ...value, naturalEnglish: event.target.value }))}
          />
        </label>
        <label className="research-ase__field">
          Intent
          <input
            value={form.intent}
            onChange={(event) => setForm((value) => ({ ...value, intent: event.target.value }))}
            placeholder="health_symptom_report"
          />
        </label>
        <label className="research-ase__field">
          Decision
          <select
            value={form.decision}
            onChange={(event) =>
              setForm((value) => ({ ...value, decision: event.target.value as ReviewDecision }))
            }
          >
            <option value="unreviewed">Unreviewed</option>
            <option value="reviewed">Reviewed</option>
            <option value="needs_second_review">Needs second review</option>
            <option value="exclude">Exclude</option>
          </select>
        </label>
        <label className="research-ase__field">
          Entities
          <textarea
            value={form.entities}
            onChange={(event) => setForm((value) => ({ ...value, entities: event.target.value }))}
            placeholder='{"symptom":"eye pain"}'
          />
        </label>
        <label className="research-ase__field">
          Ambiguity
          <textarea
            value={form.ambiguities}
            onChange={(event) => setForm((value) => ({ ...value, ambiguities: event.target.value }))}
          />
        </label>
        <label className="research-ase__field">
          Literal English
          <textarea
            value={form.literalEnglish}
            onChange={(event) => setForm((value) => ({ ...value, literalEnglish: event.target.value }))}
          />
        </label>
        <label className="research-ase__field">
          Review notes
          <textarea
            value={form.notes}
            onChange={(event) => setForm((value) => ({ ...value, notes: event.target.value }))}
          />
        </label>
      </div>

      <div className="research-ase__actions">
        <button type="button" onClick={() => setSelectedIndex((value) => Math.max(0, value - 1))}>
          <ArrowLeft className="h-4 w-4" />
          Back
        </button>
        <button type="button" onClick={() => void save(form)} disabled={saving}>
          <Save className="h-4 w-4" />
          {saving ? "Saving" : "Save"}
        </button>
        <button
          type="button"
          onClick={() => void save({ ...form, decision: "needs_second_review" }, nextIndex)}
          disabled={saving}
        >
          Second review
        </button>
        <button
          type="button"
          onClick={() => void save({ ...form, decision: "exclude" }, nextIndex)}
          disabled={saving}
        >
          Exclude
        </button>
        <button
          type="button"
          className="research-ase__primary"
          onClick={() => void save({ ...form, decision: "reviewed" }, nextIndex)}
          disabled={saving || !canAccept}
          title={canAccept ? "Mark reviewed and continue" : "Needs Twi, English meaning, and intent"}
        >
          <Check className="h-4 w-4" />
          Reviewed. Next
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </main>
  );
}
