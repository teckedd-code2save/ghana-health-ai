#!/usr/bin/env python3
"""Verify downloaded R2 artifacts and render an inspectable, offline result report."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import sys

from r2_inference import load_bundle, read, require, save, sha, safe, VARIANTS

LABELS = {"base": "Unadapted base", "r1_meaning": "R1: meaning", "r1_response": "R1: responses",
          "r2_language": "R2: language", "r2_meaning": "R2: language + meaning", "r2_response": "R2: final response stage"}


def report(bundle, results, out):
    manifest = load_bundle(bundle)
    binding = sha(bundle / "manifest.json")
    index = read(results / "results-manifest.json")
    require(index["execution_sha256"] == binding, "Result execution identity changed")
    for file, checksum in index["files"].items():
        require(sha(safe(results, file)) == checksum, "Result checksum mismatch: " + file)
    status = read(results / "status.json")
    require(status["state"] == "completed" and status["manifest_sha256"] == binding
            and status["chat_slots"] == 216 and status["language_rows"] == 2058, "Evaluation incomplete")
    summary = read(results / "summary.json")
    cases = read(bundle / "evaluation/cases.json")
    sys.path.insert(0, str(bundle / "code"))
    from build_understanding_evaluation import objective_score
    by_case, verified = {}, {}
    for variant in VARIANTS:
        records = {c["id"]: read(results / variant / "chat" / (c["id"] + ".json")) for c in cases}
        terms = [read(p) for p in sorted((results / variant / "loss").glob("*.json"))]
        require(len(terms) == 343 and len({t["id"] for t in terms}) == 343, "Loss coverage differs")
        tokens, nll = sum(t["predicted_tokens"] for t in terms), sum(t["summed_nll"] for t in terms)
        stats = summary["checkpoints"][variant]
        require(stats["language_loss"]["predicted_tokens"] == tokens
                and math.isclose(stats["language_loss"]["summed_nll"], nll, abs_tol=1e-8), "Loss aggregate differs")
        scores = [objective_score(c, records[c["id"]]) for c in cases if c["objective"]]
        require(sum(s["passed"] for s in scores) == stats["strict_objective_passed"], "Objective aggregate differs")
        require(dict(Counter(r["finish_reason"] for r in records.values())) == stats["finish_reasons"], "Stop counts differ")
        repeated = sum(r["diagnostics"]["repetition"]["4_word_ngrams"]["repeated_positions"] > 0 for r in records.values())
        require(repeated == stats["replies_with_repeated_4grams"], "Repetition aggregate differs")
        verified[variant] = {**stats, "artifact_checksums_verified": True}
        for c in cases:
            row = records[c["id"]]
            require(row["case_id"] == c["id"] and row["identity"]["variant"] == variant
                    and row["identity"]["manifest_sha256"] == binding, "Reply identity differs")
            by_case.setdefault(c["id"], {})[variant] = row
    require(not out.exists(), "Use a versioned new report directory")
    out.mkdir(parents=True)
    proof = {"verified_at": datetime.now(timezone.utc).isoformat(), "execution_sha256": binding,
             "result_manifest_sha256": sha(results / "results-manifest.json"), "verified_files": len(index["files"]),
             "native_review": "pending", "checkpoints": verified, "local_default": "r2_meaning"}
    save(out / "verification.json", proof)
    esc = lambda value: html.escape(str(value))
    rows = "".join(f'<tr><th>{esc(LABELS[v])}</th><td>{verified[v]["language_loss"]["token_weighted_mean_nll"]:.3f}</td>'
        f'<td>{verified[v]["language_loss"]["exp_mean_nll"]:.2f}</td><td>{verified[v]["strict_objective_passed"]}/8</td>'
        f'<td>{verified[v]["finish_reasons"].get("eos", 0)}/36</td><td>{verified[v]["replies_with_repeated_4grams"]}/36</td></tr>' for v in VARIANTS)
    details = []
    for c in cases:
        history = "".join(f'<p><b>{esc(m["role"])}</b> {esc(m["content"])}</p>' for m in c["messages"])
        variants = "".join(f'<section class="reply"><h3>{esc(LABELS[v])}</h3><p class="meta">'
            f'{esc(by_case[c["id"]][v]["finish_reason"])} | {esc(by_case[c["id"]][v]["origin"])}</p>'
            f'<pre>{esc(by_case[c["id"]][v]["output"])}</pre></section>' for v in VARIANTS)
        rubric = "".join(f'<li>{esc(x)}</li>' for x in c.get("expected_meaning_checks", ["No case-specific rubric was recorded for this live regression."]))
        details.append(f'<details class="case"><summary>{esc(c["id"])}</summary><div class="history">{history}</div>'
                       f'<p class="meta">Review criteria, not a human-approved score:</p><ul>{rubric}</ul><div class="replies">{variants}</div></details>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Twi R2: measured progress</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#fafbfc;color:#23262d;font:16px/1.6 system-ui,sans-serif;letter-spacing:0}}
main{{max-width:1180px;margin:auto;padding:40px 24px}}header{{border-bottom:2px solid #27344a;padding-bottom:24px}}
h1{{font-size:32px;line-height:1.2}}h2{{font-size:23px;margin-top:36px}}h3{{font-size:16px;margin:0}}a{{color:#245fb4}}
.meta{{font-size:13px;color:#5b6573}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;text-align:left}}
th,td{{padding:12px;border-bottom:1px solid #dbe0e6;white-space:nowrap}}thead{{background:#eaf0f8}}
tbody th{{font-weight:500}}.case{{border-bottom:1px solid #d2d9e3;padding:16px 0}}summary{{cursor:pointer;font-weight:600}}
.history{{border-left:3px solid #7a9cc8;padding-left:16px}}.replies{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}
.reply{{border-top:1px solid #dbe0e6;padding-top:16px;min-width:0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.6 system-ui}}
code{{overflow-wrap:anywhere}}.callout{{border-left:3px solid #bc7327;padding-left:16px}}
@media(max-width:680px){{main{{padding:24px 16px}}h1{{font-size:26px}}.replies{{grid-template-columns:1fr}}}}
@media print{{main{{max-width:none;padding:0}}.case{{break-inside:avoid}}a{{color:inherit}}}}
</style><main><header><p class="meta">Ghana Health AI | 5 November 2026 | Research evidence</p>
<h1>Twi R2: measured progress</h1><p>I added source-backed Twi language learning before meaning alignment, then tested each stage independently.</p>
<p><a href="http://127.0.0.1:7866/?__theme=light">Open the live local R2 test</a> &middot; <a href="verification.json">Verification receipt</a></p></header>
<h2>What I can demonstrate</h2><p>I have completed a reproducible three-stage LoRA run on the pinned AfriqueQwen 9B foundation, exported all three checkpoints and evaluated them alongside R1 and the unadapted base. The local page serves the actual selected checkpoint, with no external model answering on its behalf.</p>
<p>The language-plus-meaning checkpoint is the local default because it completes these development prompts without the repeated four-word phrases observed in the final response stage. This is a diagnostic choice, not a declaration of broad Twi competence.</p>
<h2>Controlled comparison</h2><div class="table"><table><thead><tr><th>Checkpoint</th><th>Twi mean NLL</th><th>Perplexity</th><th>Strict JSON tasks</th><th>End-token stops</th><th>Any repeated 4-gram</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="meta">Language loss: 343 protected source-backed validation texts, 9,315 predicted tokens per checkpoint. Lower NLL/perplexity means better next-token prediction on this subset, not verified understanding. Chat: the same 36 development prompts; 8 have literal JSON checks. End-token stops can still contain wrong answers. Repeated phrases can be legitimate, so inspect the raw replies below.</p>
<h2>What changed, and what remains</h2><ul><li>I preserved the pinned base, tokenizer, R1 meaning/response schedules and decoding. R2 adds the language stage before them; it begins from a fresh adapter, not R1's adapter.</li>
<li>The language stage improves held-out Twi token prediction. R2 meaning preserves the 8/8 strict task result while removing repeated four-word phrases in these 36 replies.</li>
<li>The response stage lowers strict JSON compliance again. Several answers contain the requested values inside forbidden Markdown fences. Its repetition and incomplete replies also need work; I am showing that stage rather than hiding it.</li>
<li>Natural multi-turn Twi conversation remains a concrete data gap. The response subset is 86 prose examples plus 220 language exercises, not 306 natural conversations.</li>
<li>Native semantic/naturalness review, a genuinely blind test, broader English-retention tests and health-specific safety assessment remain open. These results are not clinical certification or population accuracy.</li></ul>
<h2>Training and provenance</h2><ul><li>Language: 16,742 source-preserved Twi sentences/utterances: 5,385 GhanaNLP parallel sentences, 11,350 GhanaNLP speech utterances and 7 community parallel utterances. No synthetic Pristine in this run.</li>
<li>Meaning: 13,439 training views. Responses: 306 source-bound examples, two passes. English replay: approximately 20% of effective target tokens at each stage.</li>
<li>Total exposure: 31,963 sequence presentations and 1,193,890 effective target tokens over 1,999 updates. Views, presentations and unique source records are different counts.</li>
<li>Base: 8,953,803,264 frozen parameters. LoRA: 43,278,336 trainable parameters, rank 16. No new vocabulary or embedding updates.</li>
<li>Original data, split protection, saved reviews, rejected translations and held-out texts remain separate. No new training or public release occurred during evaluation.</li></ul>
<h2>Evaluation method</h2><p>I froze the 30 existing development prompts plus 6 raw live regressions before training. I use complete fixed histories and greedy non-thinking generation, 256 output tokens, no prompt repair, no fallback and no repetition guard. The comparison reuses 102 exact-protocol saved replies and generates 114 missing replies. All 216 response slots and 2,058 loss records are preserved. Human judgment is separate from exact task scoring.</p>
<p>I use explicit provenance, language-specific evidence, immutable evaluation inputs and transparent failure reporting to frame the work as equitable and trustworthy AI research. I do not present one small benchmark as universal language ability.</p>
<h2>Every development example</h2><p class="meta">Open a prompt to compare all six raw replies. Held-out language texts are deliberately not reproduced here.</p>{''.join(details)}
<h2>Evidence and next decision</h2><p>Training: <code>understanding_20261105T192103Z</code>. Evaluation: <code>{esc(status['run_id'])}</code>. Evaluation bundle: <code>{binding}</code>.</p>
<p>I can now ask a precise question: which conversation examples and response targets improve naturalness and follow-up meaning without undoing the language/meaning gains? I will use the live reviews to identify those failures before choosing the next training intervention.</p>
</main></html>'''
    (out / "index.html").write_text(page)
    save(out / "checksums.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    return {"report": str(out / "index.html"), "verified_files": len(index["files"]), "native_review": "pending"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True); p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(report(a.bundle, a.results, a.out), indent=2))
