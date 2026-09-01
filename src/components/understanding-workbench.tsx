"use client";

import { Dispatch, SetStateAction, useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Database,
  FileCheck2,
  Layers3,
  ListChecks,
  Save,
  ShieldCheck,
  Upload,
} from "lucide-react";

type ReviewDecision = "unreviewed" | "reviewed" | "needs_second_review" | "exclude";
type CorpusFilter = "priority" | "training_data" | "local_audio" | "medical" | "curated" | "all";

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

type BenchmarkRow = {
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
  corpus: {
    sourceInventory: Array<{
      id: string;
      name: string;
      role: string;
      evidence: string;
      storage: string;
      status: string;
    }>;
    storageDecisions: Array<{ material: string; systemOfRecord: string; rule: string }>;
    stages: string[];
  };
  benchmark: {
    rows: BenchmarkRow[];
    total: number;
    completed: number;
    needsSecondReview: number;
    excluded: number;
    scorecard: {
      created_at: string;
      decision_hint: string;
      scorecards: Array<{
        candidate: string;
        cases_scored: number;
        exact_cases: number;
        checks: number;
        passed: number;
        score: number;
        elapsed_seconds: number;
        critical_failures: Array<{
          id: string;
          category: string;
          text: string;
          prediction: string;
          failures: string[];
        }>;
      }>;
    } | null;
  };
  candidates: {
    rows: BenchmarkRow[];
    total: number;
    completed: number;
    withAudio: number;
    draftAnnotated: number;
    trainingReady: number;
    splits: {
      train: number;
      dev: number;
      test: number;
    };
    candidateSplits: {
      train: number;
      dev: number;
      test: number;
    };
    readiness: {
      ready: boolean;
      required_passed: number;
      required_total: number;
      checks: Array<{
        id: string;
        label: string;
        passed: boolean;
        value: number | string | boolean;
        required: number | string | boolean;
        severity: "required" | "warning";
      }>;
    };
  };
};

const corpusFilterOptions: Array<[CorpusFilter, string]> = [
  ["training_data", "Training data"],
  ["medical", "Medical"],
  ["local_audio", "Local audio"],
  ["curated", "Curated"],
  ["all", "All"],
];

const emptyReview = (row?: BenchmarkRow): Review => ({
  id: row?.id ?? "",
  normalizedTwi: row?.modelProposal?.normalized_twi ?? "",
  naturalEnglish: row?.modelProposal?.natural_english ?? "",
  literalEnglish: row?.modelProposal?.literal_english ?? "",
  intent: row?.modelProposal?.intent ?? "",
  entities: row?.modelProposal?.entities ?? "",
  ambiguities: row?.modelProposal?.ambiguities ?? "",
  decision: "unreviewed",
  notes: "",
});

function reviewRank(row: BenchmarkRow) {
  const reviewed = row.review?.decision === "reviewed" ? 100 : 0;
  const excluded = row.review?.decision === "exclude" ? 120 : 0;
  const secondReview = row.review?.decision === "needs_second_review" ? 80 : 0;
  const health = row.category.includes("health") ? -20 : 0;
  const audio = row.audioArtifactId ? -10 : 0;
  const local = row.source === "local_recording" ? -8 : 0;
  const splitCoverage = row.trainingSplit === "test" ? -18 : row.trainingSplit === "dev" ? -16 : 0;
  return reviewed + excluded + secondReview + health + audio + local + splitCoverage;
}

function fillFromDraft(row: BenchmarkRow, prior?: Review): Review {
  return {
    id: row.id,
    normalizedTwi: row.modelProposal?.normalized_twi || prior?.normalizedTwi || row.text,
    naturalEnglish: row.modelProposal?.natural_english || prior?.naturalEnglish || "",
    literalEnglish: row.modelProposal?.literal_english || prior?.literalEnglish || "",
    intent: row.modelProposal?.intent || prior?.intent || "",
    entities: row.modelProposal?.entities || prior?.entities || "",
    ambiguities: row.modelProposal?.ambiguities || prior?.ambiguities || "",
    decision: prior?.decision ?? "unreviewed",
    notes: prior?.notes ?? "",
  };
}

export function UnderstandingWorkbench() {
  const [payload, setPayload] = useState<WorkbenchPayload | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [tab, setTab] = useState<"sources" | "benchmark" | "corpus" | "review">("sources");
  const [reviewMode, setReviewMode] = useState<"benchmark" | "corpus">("corpus");
  const [corpusFilter, setCorpusFilter] = useState<CorpusFilter>("training_data");

  async function load() {
    const res = await fetch("/api/research/understanding", { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) {
      setError(data.error ?? "Research workspace is unavailable.");
      return;
    }
    setPayload(data);
  }

  useEffect(() => {
    let cancelled = false;
    async function run() {
      const res = await fetch("/api/research/understanding", { cache: "no-store" });
      const data = await res.json();
      if (cancelled) return;
      if (!res.ok) {
        setError(data.error ?? "Research workspace is unavailable.");
        return;
      }
      setPayload(data);
    }
    void run();
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredCorpusRows = useMemo(() => {
    const corpusRows = payload?.candidates.rows ?? [];
    const filtered = corpusRows.filter((row) => {
      if (corpusFilter === "all" || corpusFilter === "priority") return true;
      if (corpusFilter === "training_data") {
        return row.source === "waxal" || row.source === "ghana_nlp_speech";
      }
      if (corpusFilter === "local_audio") return row.source === "local_recording";
      if (corpusFilter === "medical") return row.source === "medical_response_seed";
      if (corpusFilter === "curated") return row.source === "curated_prompt";
      return true;
    });
    return filtered.sort((a, b) => reviewRank(a) - reviewRank(b) || a.id.localeCompare(b.id));
  }, [payload, corpusFilter]);
  const rows = useMemo(
    () =>
      reviewMode === "corpus"
        ? filteredCorpusRows
        : [...(payload?.benchmark.rows ?? [])].sort(
            (a, b) => reviewRank(a) - reviewRank(b) || a.id.localeCompare(b.id),
          ),
    [filteredCorpusRows, payload, reviewMode],
  );
  const corpusReviewRows = useMemo(
    () => filteredCorpusRows,
    [filteredCorpusRows],
  );
  const selected = rows[selectedIndex];
  const corpusReviewSummary = useMemo(() => {
    const corpusRows = payload?.candidates.rows ?? [];
    const accepted = corpusRows.filter((row) => row.review?.decision === "reviewed").length;
    const secondReview = corpusRows.filter((row) => row.review?.decision === "needs_second_review").length;
    const excluded = corpusRows.filter((row) => row.review?.decision === "exclude").length;
    return {
      accepted,
      secondReview,
      excluded,
      unreviewed: Math.max(0, corpusRows.length - accepted - secondReview - excluded),
    };
  }, [payload]);
  const nextCorpusReviewIndex = useMemo(() => {
    const index = corpusReviewRows.findIndex((row) => !row.review || row.review.decision === "unreviewed");
    return index >= 0 ? index : 0;
  }, [corpusReviewRows]);
  const progress = payload
    ? Math.round((payload.candidates.trainingReady / Math.max(payload.candidates.total, 1)) * 100)
    : 0;

  const benchmarkCategoryCounts = useMemo(() => {
    const counts = new Map<string, number>();
    (payload?.benchmark.rows ?? []).forEach((row) =>
      counts.set(row.category, (counts.get(row.category) ?? 0) + 1),
    );
    return Array.from(counts.entries());
  }, [payload]);

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
    setUploadStatus(
      `Imported ${data.imported} rows, skipped ${data.skipped}, accepted ${data.accepted}. Readiness ${data.readiness.required_passed}/${data.readiness.required_total}.`,
    );
    await load();
  }

  return (
    <section className="research-ase">
      <header className="research-ase__hero">
        <div>
          <p className="research-ase__eyebrow">Akan Speech Evidence</p>
          <h1>Understanding corpus workbench</h1>
          <p>
            Build the real corpus from licensed datasets and consented records. The 50 synthetic rows
            are only a small benchmark for comparing candidate models.
          </p>
        </div>
        <div className="research-ase__status">
          <span>{payload ? `${payload.benchmark.total} benchmark probes` : "Loading"}</span>
          <strong>{payload ? `${progress}% training-ready` : "..."}</strong>
        </div>
      </header>

      <nav className="research-ase__tabs" aria-label="Research workspace sections">
        <button className={tab === "sources" ? "is-active" : ""} onClick={() => setTab("sources")}>
          <Database className="h-4 w-4" />
          Sources
        </button>
        <button className={tab === "benchmark" ? "is-active" : ""} onClick={() => setTab("benchmark")}>
          <Layers3 className="h-4 w-4" />
          Benchmark
        </button>
        <button className={tab === "corpus" ? "is-active" : ""} onClick={() => setTab("corpus")}>
          <ListChecks className="h-4 w-4" />
          Corpus
        </button>
        <button className={tab === "review" ? "is-active" : ""} onClick={() => setTab("review")}>
          <FileCheck2 className="h-4 w-4" />
          Review
        </button>
      </nav>

      {error && <p className="research-ase__error">{error}</p>}

      {tab === "sources" && payload && (
        <div className="research-ase__grid">
          <section className="research-ase__panel research-ase__panel--wide">
            <div className="research-ase__panel-head">
              <ShieldCheck className="h-5 w-5" />
              <div>
                <h2>Corpus route</h2>
                <p>No model trains from drafts. Every row must keep provenance, review state, and split safety.</p>
              </div>
            </div>
            <ol className="research-ase__steps">
              {payload.corpus.stages.map((stage) => (
                <li key={stage}>{stage}</li>
              ))}
            </ol>
          </section>

          <section className="research-ase__panel">
            <h2>Storage decisions</h2>
            <div className="research-ase__table">
              {payload.corpus.storageDecisions.map((item) => (
                <div key={item.material}>
                  <strong>{item.material}</strong>
                  <span>{item.systemOfRecord}</span>
                  <p>{item.rule}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="research-ase__sources">
            {payload.corpus.sourceInventory.map((source) => (
              <article key={source.id}>
                <span>{source.evidence}</span>
                <h3>{source.name}</h3>
                <p>{source.role}</p>
                <small>{source.storage}</small>
                <em>{source.status}</em>
              </article>
            ))}
          </section>
        </div>
      )}

      {tab === "benchmark" && payload && (
        <section className="research-ase__panel">
          <div className="research-ase__panel-head">
            <Layers3 className="h-5 w-5" />
            <div>
              <h2>Benchmark probes</h2>
              <p>
                These rows measure model behaviour. They are not the training corpus and not gold
                labels until reviewed.
              </p>
            </div>
          </div>
          <div className="research-ase__metrics">
            <span>{payload.benchmark.completed} reviewed</span>
            <span>{payload.benchmark.needsSecondReview} need second review</span>
            <span>{payload.benchmark.excluded} excluded</span>
          </div>
          {payload.benchmark.scorecard && (
            <div className="research-ase__scorecards">
              {payload.benchmark.scorecard.scorecards.map((scorecard, index) => (
                <article key={scorecard.candidate} className={index === 0 ? "is-leading" : ""}>
                  <div>
                    <span>{index === 0 ? "Current best fit" : "Candidate"}</span>
                    <h3>{scorecard.candidate}</h3>
                  </div>
                  <strong>{Math.round(scorecard.score * 100)}%</strong>
                  <p>
                    {scorecard.exact_cases}/{scorecard.cases_scored} exact cases · {scorecard.passed}/
                    {scorecard.checks} meaning checks · {scorecard.elapsed_seconds}s
                  </p>
                  {scorecard.critical_failures.length > 0 && (
                    <small>{scorecard.critical_failures.length} critical failures still need review.</small>
                  )}
                </article>
              ))}
              <p className="research-ase__hint">{payload.benchmark.scorecard.decision_hint}</p>
            </div>
          )}
          <div className="research-ase__categories">
            {benchmarkCategoryCounts.map(([category, count]) => (
              <span key={category}>
                {category} <strong>{count}</strong>
              </span>
            ))}
          </div>
        </section>
      )}

      {tab === "corpus" && payload && (
        <section className="research-ase__panel">
          <div className="research-ase__panel-head">
            <ListChecks className="h-5 w-5" />
            <div>
              <h2>Corpus candidates</h2>
              <p>
                These are dataset-derived and consent-scoped candidates. Model-populated fields are
                drafts until you review and correct them.
              </p>
            </div>
          </div>
          <div className="research-ase__metrics">
            <span>{payload.candidates.total} candidate rows</span>
            <span>{payload.candidates.withAudio} with audio references</span>
            <span>{payload.candidates.draftAnnotated} with draft annotations</span>
            <span>{payload.candidates.completed} reviewed</span>
            <span>{payload.candidates.trainingReady} training-ready</span>
            <span>{corpusReviewSummary.unreviewed} left to review</span>
            <span>
              readiness {payload.candidates.readiness.required_passed}/
              {payload.candidates.readiness.required_total}
            </span>
            <span>
              train/dev/test {payload.candidates.splits.train}/{payload.candidates.splits.dev}/
              {payload.candidates.splits.test}
            </span>
            <span>
              available {payload.candidates.candidateSplits.train}/
              {payload.candidates.candidateSplits.dev}/{payload.candidates.candidateSplits.test}
            </span>
          </div>
          <div className="research-ase__readiness">
            {payload.candidates.readiness.checks.map((check) => (
              <span
                key={check.id}
                className={[
                  check.passed ? "is-passed" : "is-failed",
                  check.severity === "warning" ? "is-warning" : "",
                ].join(" ")}
              >
                {check.label}: {String(check.value)}
              </span>
            ))}
          </div>
          <div className="research-ase__quick-actions">
            <button
              type="button"
              onClick={() => {
                setReviewMode("corpus");
                setCorpusFilter("priority");
                setSelectedIndex(nextCorpusReviewIndex);
                setTab("review");
              }}
            >
              Review next training row
            </button>
            <a
              className="research-ase__export-link"
              href="/api/research/understanding/export"
              target="_blank"
              rel="noreferrer"
            >
              Open training export
            </a>
            <a
              className="research-ase__export-link"
              href="/api/research/understanding/review-sheet"
              target="_blank"
              rel="noreferrer"
            >
              Download review sheet
            </a>
            <a
              className="research-ase__export-link"
              href="/api/research/understanding/review-sheet?scope=minimum-training"
              target="_blank"
              rel="noreferrer"
            >
              Download 20-row training pack
            </a>
            <a
              className="research-ase__export-link"
              href="/api/research/understanding/review-sheet?scope=minimum-training&prefill=draft"
              target="_blank"
              rel="noreferrer"
            >
              Download assisted pack
            </a>
            <label className="research-ase__upload">
              <Upload className="h-4 w-4" />
              {uploading ? "Importing..." : "Upload reviewed CSV"}
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
          <div className="research-ase__filters" aria-label="Corpus row filters">
            {corpusFilterOptions.map(([value, label]) => (
              <button
                key={value}
                type="button"
                className={corpusFilter === value ? "is-active" : ""}
                onClick={() => setCorpusFilter(value as CorpusFilter)}
              >
                {label}
              </button>
            ))}
          </div>
          {uploadStatus && <p className="research-ase__success">{uploadStatus}</p>}
          <div className="research-ase__candidate-list">
            {corpusReviewRows.slice(0, 24).map((row, index) => (
              <button
                key={row.id}
                type="button"
                onClick={() => {
                  setReviewMode("corpus");
                  setSelectedIndex(index);
                  setTab("review");
                }}
              >
                <strong>{row.text}</strong>
                <span>
                  {row.category} · {row.split ?? "unknown split"} · {row.modelProposal?.status ?? "no draft"}
                  {row.trainingSplit ? ` · ${row.trainingSplit}` : ""}
                </span>
              </button>
            ))}
          </div>
        </section>
      )}

      {tab === "review" && payload && selected && (
        <ReviewEditor
          key={selected.id}
          rows={rows}
          selected={selected}
          selectedIndex={selectedIndex}
          reviewMode={reviewMode}
          trainingReady={payload.candidates.trainingReady}
          corpusReviewSummary={corpusReviewSummary}
          setReviewMode={setReviewMode}
          benchmarkTotal={payload.benchmark.total}
          corpusTotal={payload.candidates.total}
          saving={saving}
          setSelectedIndex={setSelectedIndex}
          save={save}
        />
      )}
    </section>
  );
}

function ReviewEditor({
  rows,
  selected,
  selectedIndex,
  reviewMode,
  trainingReady,
  corpusReviewSummary,
  setReviewMode,
  benchmarkTotal,
  corpusTotal,
  saving,
  setSelectedIndex,
  save,
}: {
  rows: BenchmarkRow[];
  selected: BenchmarkRow;
  selectedIndex: number;
  reviewMode: "benchmark" | "corpus";
  trainingReady: number;
  corpusReviewSummary: {
    accepted: number;
    secondReview: number;
    excluded: number;
    unreviewed: number;
  };
  setReviewMode: Dispatch<SetStateAction<"benchmark" | "corpus">>;
  benchmarkTotal: number;
  corpusTotal: number;
  saving: boolean;
  setSelectedIndex: Dispatch<SetStateAction<number>>;
  save: (review: Review, nextIndex?: number) => Promise<void>;
}) {
  const [form, setForm] = useState<Review>(selected.review ?? emptyReview(selected));
  const canAccept =
    form.normalizedTwi.trim().length > 0 &&
    form.naturalEnglish.trim().length > 0 &&
    form.intent.trim().length > 0;
  const nextIndex = Math.min(rows.length - 1, selectedIndex + 1);

  return (
    <section className="research-ase__review">
          <div className="research-ase__review-topbar">
            <div className="research-ase__review-switch">
            <button
              type="button"
              className={reviewMode === "corpus" ? "is-active" : ""}
              onClick={() => {
                setReviewMode("corpus");
                setSelectedIndex(0);
              }}
            >
              Corpus candidates <span>{corpusTotal}</span>
            </button>
            <button
              type="button"
              className={reviewMode === "benchmark" ? "is-active" : ""}
              onClick={() => {
                setReviewMode("benchmark");
                setSelectedIndex(0);
              }}
            >
              Benchmark probes <span>{benchmarkTotal}</span>
            </button>
            </div>
            {reviewMode === "corpus" && (
              <div className="research-ase__review-progress">
                <span>
                  Row {selectedIndex + 1}/{rows.length}
                </span>
                <span>{trainingReady} ready</span>
                <span>{corpusReviewSummary.unreviewed} unreviewed</span>
              </div>
            )}
          </div>
          {reviewMode === "benchmark" && (
            <div className="research-ase__review-progress">
            <span>
              Row {selectedIndex + 1}/{rows.length}
            </span>
            <span>{trainingReady} training-ready</span>
            <span>{corpusReviewSummary.unreviewed} unreviewed</span>
            <span>{corpusReviewSummary.secondReview} second review</span>
            <span>{corpusReviewSummary.excluded} excluded</span>
            </div>
          )}
          <aside className="research-ase__queue" aria-label="Benchmark row queue">
            {rows.slice(0, 120).map((row, index) => (
              <button
                key={row.id}
                className={index === selectedIndex ? "is-active" : ""}
                onClick={() => setSelectedIndex(index)}
              >
                <span>{row.text || row.id}</span>
                <small>
                  {row.source ?? row.kind} · {row.trainingSplit ?? row.split ?? "split?"}
                </small>
              </button>
            ))}
          </aside>

          <div className="research-ase__editor">
            <div className="research-ase__prompt">
              <span>
                {selected.category} · {selected.source ?? selected.kind} · {selected.trainingSplit ?? selected.split ?? "split?"}
              </span>
              <p>{selected.text}</p>
              <details>
                <summary>Source details</summary>
                <small>
                  {selected.kind === "corpus"
                    ? `${selected.sourceRecordId ?? selected.id} · ${selected.consentScope ?? "unknown consent"} · ${selected.speakerId ?? "unknown speaker"}`
                    : "Synthetic benchmark probe"}
                </small>
                {selected.audioArtifactId && <small>{selected.audioArtifactId}</small>}
              </details>
              <div className="research-ase__prompt-actions">
                <button
                  type="button"
                  onClick={() => setForm((value) => ({ ...value, normalizedTwi: selected.text }))}
                >
                  Use transcript
                </button>
                {selected.modelProposal?.status === "draft" && (
                  <button type="button" onClick={() => setForm((value) => fillFromDraft(selected, value))}>
                    Use draft
                  </button>
                )}
              </div>
            </div>

            <label>
              Normalized Twi
              <textarea
                value={form.normalizedTwi}
                onChange={(event) => setForm((value) => ({ ...value, normalizedTwi: event.target.value }))}
              />
            </label>
            <label>
              Faithful English meaning
              <textarea
                value={form.naturalEnglish}
                onChange={(event) => setForm((value) => ({ ...value, naturalEnglish: event.target.value }))}
              />
            </label>
            <label>
              Literal English, if helpful
              <textarea
                value={form.literalEnglish}
                onChange={(event) => setForm((value) => ({ ...value, literalEnglish: event.target.value }))}
              />
            </label>
            <div className="research-ase__two">
              <label>
                Intent
                <input
                  value={form.intent}
                  onChange={(event) => setForm((value) => ({ ...value, intent: event.target.value }))}
                  placeholder="report_symptom"
                />
              </label>
              <label>
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
            </div>
            <label>
              Entities
              <textarea
                value={form.entities}
                onChange={(event) => setForm((value) => ({ ...value, entities: event.target.value }))}
                placeholder="symptom=headache; duration=since yesterday"
              />
            </label>
            <label>
              Ambiguity or uncertainty
              <textarea
                value={form.ambiguities}
                onChange={(event) => setForm((value) => ({ ...value, ambiguities: event.target.value }))}
              />
            </label>
            <label>
              Review notes
              <textarea
                value={form.notes}
                onChange={(event) => setForm((value) => ({ ...value, notes: event.target.value }))}
              />
            </label>

            <div className="research-ase__actions">
              <button type="button" onClick={() => setSelectedIndex((value) => Math.max(0, value - 1))}>
                <ArrowLeft className="h-4 w-4" />
                Previous
              </button>
              <button type="button" onClick={() => void save(form)} disabled={saving}>
                <Save className="h-4 w-4" />
                {saving ? "Saving" : "Save"}
              </button>
              <button
                type="button"
                onClick={() =>
                  void save(
                    {
                      ...form,
                      decision: "needs_second_review",
                    },
                    nextIndex,
                  )
                }
                disabled={saving}
              >
                Second review
              </button>
              <button
                type="button"
                onClick={() =>
                  void save(
                    {
                      ...form,
                      decision: "exclude",
                    },
                    nextIndex,
                  )
                }
                disabled={saving}
              >
                Exclude
              </button>
              <button
                type="button"
                className="research-ase__primary"
                onClick={() =>
                  void save(
                    {
                      ...form,
                      decision: "reviewed",
                    },
                    nextIndex,
                  )
                }
                disabled={saving || !canAccept}
                title={canAccept ? "Mark reviewed and continue" : "Needs Twi, English meaning, and intent"}
              >
                <Check className="h-4 w-4" />
                Accept and next
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          </div>
    </section>
  );
}
