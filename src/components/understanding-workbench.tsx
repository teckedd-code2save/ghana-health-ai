"use client";

import { Dispatch, SetStateAction, useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Database,
  FileCheck2,
  Layers3,
  Save,
  ShieldCheck,
} from "lucide-react";

type ReviewDecision = "unreviewed" | "reviewed" | "needs_second_review" | "exclude";

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
  id: string;
  category: string;
  text: string;
  review_status: string;
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
  };
};

const emptyReview = (row?: BenchmarkRow): Review => ({
  id: row?.id ?? "",
  normalizedTwi: "",
  naturalEnglish: "",
  literalEnglish: "",
  intent: "",
  entities: "",
  ambiguities: "",
  decision: "unreviewed",
  notes: "",
});

export function UnderstandingWorkbench() {
  const [payload, setPayload] = useState<WorkbenchPayload | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [tab, setTab] = useState<"sources" | "benchmark" | "review">("sources");

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

  const rows = useMemo(() => payload?.benchmark.rows ?? [], [payload]);
  const selected = rows[selectedIndex];
  const progress = payload
    ? Math.round((payload.benchmark.completed / Math.max(payload.benchmark.total, 1)) * 100)
    : 0;

  const categoryCounts = useMemo(() => {
    const counts = new Map<string, number>();
    rows.forEach((row) => counts.set(row.category, (counts.get(row.category) ?? 0) + 1));
    return Array.from(counts.entries());
  }, [rows]);

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
          <strong>{payload ? `${progress}% reviewed` : "..."}</strong>
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
          <div className="research-ase__categories">
            {categoryCounts.map(([category, count]) => (
              <span key={category}>
                {category} <strong>{count}</strong>
              </span>
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
  saving,
  setSelectedIndex,
  save,
}: {
  rows: BenchmarkRow[];
  selected: BenchmarkRow;
  selectedIndex: number;
  saving: boolean;
  setSelectedIndex: Dispatch<SetStateAction<number>>;
  save: (review: Review, nextIndex?: number) => Promise<void>;
}) {
  const [form, setForm] = useState<Review>(selected.review ?? emptyReview(selected));

  return (
    <section className="research-ase__review">
          <aside className="research-ase__queue" aria-label="Benchmark row queue">
            {rows.map((row, index) => (
              <button
                key={row.id}
                className={index === selectedIndex ? "is-active" : ""}
                onClick={() => setSelectedIndex(index)}
              >
                <span>{row.id}</span>
                <small>{row.review?.decision ?? "unreviewed"}</small>
              </button>
            ))}
          </aside>

          <div className="research-ase__editor">
            <div className="research-ase__prompt">
              <span>{selected.category}</span>
              <p>{selected.text}</p>
              <button
                type="button"
                onClick={() => setForm((value) => ({ ...value, normalizedTwi: selected.text }))}
              >
                Use as normalized Twi
              </button>
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
                className="research-ase__primary"
                onClick={() => void save(form, selectedIndex + 1)}
                disabled={saving}
              >
                <Check className="h-4 w-4" />
                Save and next
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          </div>
    </section>
  );
}
