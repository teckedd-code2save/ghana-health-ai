"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Check, Flag, Save, X } from "lucide-react";
import styles from "./corpus-release-workbench.module.css";

type Row = { id: string; source_hash: string; original: string; reference_english?: string; reference_answer?: string;
  source: string; language: string; structured?: unknown; messages?: Array<{role: string; content: string}>;
  context?: string; alternatives?: Array<{ model: string; text: string; fields?: Record<string, unknown> }>;
  reference_analysis?: {id: string; analysis: {record_function: string; conversational_intent: {label: string; quote: string} | null;
    entities: unknown[]; meaning_features: Record<string, unknown>; ambiguity: unknown[]} | null};
  annotation_status?: string;
  review?: {selected: number; correction: string; notes: string; decision: string; correction_field?: string; issue?: string; reference_analysis_id?: string} };

export function CorpusReleaseWorkbench({ release, endpoint = "/api/research/corpus-release", localSaveLabel = "Saved on this device" }: { release: string; endpoint?: string; localSaveLabel?: string }) {
  const [collection, setCollection] = useState("meaning");
  const [offset, setOffset] = useState(0);
  const [row, setRow] = useState<Row | null>(null);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState(-1);
  const [correction, setCorrection] = useState("");
  const [notes, setNotes] = useState("");
  const [correctionField, setCorrectionField] = useState("meaning_english");
  const [issue, setIssue] = useState("none");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [restored, setRestored] = useState(false);
  const pendingSave = useRef<{ fingerprint: string; id: string } | null>(null);
  const draftKey = (value: Row) => `corpus-draft:${release}:${value.id}:${value.source_hash}`;

  useEffect(() => {
    try {
      const url = new URL(window.location.href);
      const requested = url.searchParams.get("collection");
      const nav = requested ? {collection: requested, offset: Number(url.searchParams.get("offset") || 0)} :
        JSON.parse(localStorage.getItem(`corpus-position:${release}`) || "null");
      if (nav && ["meaning", "language", "conversation", "health", "teacher"].includes(nav.collection) && Number.isInteger(nav.offset) && nav.offset >= 0) {
        setCollection(nav.collection); setOffset(nav.offset);
      }
    } catch { /* Invalid browser state never changes the source ledger. */ }
    setRestored(true);
  }, [release]);

  function remember(change: Record<string, string | number>) {
    if (!row) return;
    try { localStorage.setItem(draftKey(row), JSON.stringify({ selected, correction, notes, correction_field: correctionField, issue,
      reference_analysis_id: row.reference_analysis?.id, ...change })); }
    catch { setNotice("Browser storage is unavailable. Save before leaving this sample."); }
  }

  useEffect(() => {
    if (!restored) return;
    const controller = new AbortController();
    try { localStorage.setItem(`corpus-position:${release}`, JSON.stringify({ collection, offset })); } catch { /* Source access does not require browser storage. */ }
    const location = new URL(window.location.href);
    location.searchParams.set("collection", collection); location.searchParams.set("offset", String(offset));
    window.history.replaceState(null, "", location);
    setLoading(true); setRow(null); setNotice("");
    fetch(`${endpoint}?${new URLSearchParams({ release, collection, offset: String(offset) })}`, { signal: controller.signal })
      .then(async (response) => {
        const value = await response.json();
        if (!response.ok || !value.ready) throw new Error(value.error || "This release is still being prepared.");
        const next = value.rows[0] as Row | undefined;
        let draft: Row["review"] | null = null;
        try { if (next) draft = JSON.parse(localStorage.getItem(draftKey(next)) || "null"); } catch { /* Keep the persisted review available. */ }
        const editing = draft ?? next?.review;
        const previousSelection = editing?.selected ?? -1;
        const sameAnnotation = editing?.reference_analysis_id === next?.reference_analysis?.id;
        setRow(next ?? null); setTotal(value.total);
        setSelected(sameAnnotation && previousSelection < (next?.alternatives?.length ?? 0) ? previousSelection : -1);
        setCorrection(editing?.correction ?? ""); setNotes(editing?.notes ?? "");
        setCorrectionField(next?.reference_analysis && !sameAnnotation && !editing?.correction ? "structured" :
          editing?.correction_field ?? (next?.reference_analysis ? "structured" : "meaning_english"));
        setIssue(editing?.issue ?? "none");
        if (draft) setNotice("Unsaved correction restored");
        else if (next?.review) setNotice(next.reference_analysis && next.review.reference_analysis_id !== next.reference_analysis.id
          ? "Source previously reviewed; new analysis" : "Previously reviewed");
        else if (!next) setNotice("No samples in this collection.");
        else if (next.annotation_status) setNotice(next.annotation_status);
      }).catch((error: Error) => { if (error.name !== "AbortError") setNotice(error.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  // draftKey depends only on release and the fetched immutable source identity.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [release, collection, offset, restored, endpoint]);

  async function save(decision: string, advance = false) {
    if (!row || busy) return;
    setBusy(true); setNotice("");
    try {
      const payload = { release, collection, offset, id: row.id, source_hash: row.source_hash, selected, correction, notes, decision, correction_field: correctionField, issue,
        ...(row.reference_analysis ? { reference_analysis_id: row.reference_analysis.id } : {}) };
      const fingerprint = JSON.stringify(payload);
      if (pendingSave.current?.fingerprint !== fingerprint) pendingSave.current = { fingerprint, id: crypto.randomUUID() };
      const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, request_id: pendingSave.current.id }) });
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || "Review was not saved.");
      setNotice(value.storage === "local_and_postgres" ? "Saved" : localSaveLabel);
      try { localStorage.removeItem(draftKey(row)); } catch { /* The server still has the saved correction. */ }
      if (advance && offset + 1 < total) setOffset(offset + 1);
    } catch (error) { setNotice(error instanceof Error ? error.message : "Review was not saved."); }
    finally { setBusy(false); }
  }

  return <main className={styles.page}>
    <header className={styles.header}><h1>Corpus review</h1>
      <select aria-label="Collection" value={collection} disabled={busy} onChange={(event) => { setCollection(event.target.value); setOffset(0); }}>
        <option value="meaning">Meaning and alignment</option><option value="language">Source text</option>
        <option value="conversation">Conversations</option><option value="health">Health</option><option value="teacher">Annotation checks</option>
      </select>
      <div className={styles.navigation}><button title="Previous sample" aria-label="Previous sample" disabled={offset === 0 || busy} onClick={() => setOffset(offset - 1)}><ArrowLeft size={19} /></button>
        <span>{total ? `${offset + 1} / ${total.toLocaleString()}` : ""}</span>
        <button title="Next sample" aria-label="Next sample" disabled={offset + 1 >= total || busy} onClick={() => setOffset(offset + 1)}><ArrowRight size={19} /></button></div>
    </header>
    <p className={styles.notice} role="status">{loading ? "Loading sample..." : notice}</p>
    {row && <>
      <section className={styles.source}><div className={styles.meta}>{row.source} · {row.language}</div><p>{row.original}</p>
        {row.reference_answer && <><h2>Reference answer</h2><p>{row.reference_answer}</p></>}
        {row.messages && <details><summary>Complete conversation</summary>{row.messages.map((m, i) => <p key={i}><strong>{m.role}</strong><br />{m.content}</p>)}</details>}
        {row.context && <details><summary>Source passage</summary><p>{row.context}</p></details>}
      </section>
      {row.reference_english && <section className={styles.reference}><h2>Existing English reference</h2><p>{row.reference_english}</p></section>}
      {Boolean(row.structured) && <section className={styles.reference}><h2>Source annotation</h2><Fields value={row.structured} /></section>}
      {row.reference_analysis?.analysis && <section className={styles.reference}><h2>English-reference analysis</h2>
        <p>{row.reference_analysis.analysis.record_function.replaceAll("_", " ")}
          {row.reference_analysis.analysis.conversational_intent ? ` · ${row.reference_analysis.analysis.conversational_intent.label}` :
            row.reference_analysis.analysis.record_function === "uncertain" ? " · Intent unresolved" : " · No direct request"}</p>
        <details><summary>Entities and meaning evidence</summary><ReferenceFields value={row.reference_analysis.analysis} /></details>
      </section>}
      {!!row.alternatives?.length && <div className={styles.alternatives}>{row.alternatives.slice(0, 2).map((option, i) => <label key={i} className={selected === i ? styles.selected : ""}>
        <span><input type="radio" name="candidate" checked={selected === i} onChange={() => { remember({ selected: i }); setSelected(i); }} />{option.model}</span><p>{option.text}</p>
        {option.fields && <details><summary>Meaning and intent</summary>{option.fields.analysis ?
          <ReferenceFields value={option.fields.analysis} checks={option.fields.automatic_checks} /> : <Fields value={option.fields} />}</details>}</label>)}</div>}
      <section className={styles.edits}>
        <label>Correction for<select aria-label="Correction for" value={correctionField} onChange={(e) => { remember({ correction_field: e.target.value }); setCorrectionField(e.target.value); }}>
          <option value="meaning_english">English meaning</option><option value="suggested_normalization">Suggested Twi spelling</option><option value="reference_answer">Answer</option><option value="conversation">Conversation</option><option value="structured">Intent or entities</option>
        </select></label>
        <label>Issue<select aria-label="Issue" value={issue} onChange={(e) => { remember({ issue: e.target.value }); setIssue(e.target.value); }}>
          <option value="none">No issue selected</option><option value="meaning">Wrong meaning</option><option value="negation">Negation</option><option value="experiencer">Who is affected</option><option value="quantity_time">Quantity or time</option><option value="naturalness">Naturalness</option><option value="unsupported_claim">Unsupported claim</option><option value="other">Other</option>
        </select></label>
        <label>Correction <textarea value={correction} onChange={(e) => { remember({ correction: e.target.value }); setCorrection(e.target.value); }} rows={3} /></label>
        <label>Review note <textarea value={notes} onChange={(e) => { remember({ notes: e.target.value }); setNotes(e.target.value); }} rows={2} /></label></section>
      <footer className={styles.footer}>
        <button title="Exclude sample" onClick={() => save("exclude", true)} disabled={busy}><X size={17} />Exclude</button>
        <button title="Needs another review" onClick={() => save("needs_second_review", true)} disabled={busy}><Flag size={17} />Flag</button>
        <button title="Save correction" aria-label="Save correction" onClick={() => save("needs_second_review")} disabled={busy}><Save size={17} />Save</button>
        <button className={styles.primary} onClick={() => save("reviewed", true)} disabled={busy}><Check size={17} />Approve and next</button>
      </footer>
    </>}
  </main>;
}

function ReferenceFields({ value, checks }: { value: unknown; checks?: unknown }) {
  if (!value || typeof value !== "object") return <span>Not annotated</span>;
  const data = value as Record<string, unknown>;
  const intent = data.conversational_intent as {label?: string; quote?: string} | null;
  const features = (data.meaning_features || {}) as Record<string, unknown>;
  const list = (items: unknown) => !Array.isArray(items) ? "Not annotated" : items.length ? <ul>{items.map((item, index) => {
    if (item && typeof item === "object") {
      const evidence = item as {label?: string; quote?: string};
      return <li key={index}><strong>{evidence.label}</strong>: {evidence.quote}</li>;
    }
    return <li key={index}>{String(item)}</li>;
  })}</ul> : "None reported";
  const failures = Array.isArray(checks) ? checks.filter(item => typeof item === "string" && !item.startsWith("Passed format")) : [];
  return <>
    <dl className={styles.analysis}>
      <div><dt>Request</dt><dd>{!("conversational_intent" in data) ? "Not annotated" : intent ? `${intent.label}: ${intent.quote}` : data.record_function === "uncertain" ? "Unresolved" : "No direct request"}</dd></div>
      <div><dt>Entities</dt><dd>{list(data.entities)}</dd></div>
      {[["negation", "Negation"], ["time", "Time"], ["quantities", "Amounts"], ["uncertainty", "Uncertainty"], ["experiencer", "Who is affected"]].map(([key, label]) =>
        <div key={key}><dt>{label}</dt><dd>{key in features ? list(features[key]) : "Not annotated"}</dd></div>)}
      <div><dt>Ambiguity</dt><dd>{list(data.ambiguity)}</dd></div>
    </dl>
    {failures.length > 0 && <p className={styles.annotationWarning}>{failures.join("; ")}</p>}
  </>;
}

function Fields({ value }: { value: unknown }) {
  if (value == null) return <span>Not annotated</span>;
  if (typeof value !== "object") return <span>{String(value)}</span>;
  if (Array.isArray(value)) return value.length ? <ul>{value.map((v, i) => <li key={i}><Fields value={v} /></li>)}</ul> : <span>None listed</span>;
  return <dl>{Object.entries(value).map(([key, item]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd><Fields value={item} /></dd></div>)}</dl>;
}
