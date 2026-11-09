#!/usr/bin/env python3
"""Render collected R4 exports offline; write private HTML only under ignored tmp/."""
import argparse
import base64
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KIND = "development_not_blind_final"
VARIANTS = ("base", "r3_response", "r4_response")
OPTIONAL = "r4_pass1"
LABELS = {"base": "Base", "r3_response": "R3 response", "r4_response": "R4 response", OPTIONAL: "R4 pass1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def number(value):
        value = float(value)
        require(math.isfinite(value), "Nonfinite JSON number")
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=number)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(folder, name):
    relative = Path(name)
    require(name and not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe artifact path")
    path = folder / relative
    require(path.resolve().is_relative_to(folder) and path.resolve() != folder
            and not any(p.is_symlink() for p in (path, *path.parents) if p != folder and p.is_relative_to(folder)),
            "Linked or escaping artifact path")
    require(path.is_file(), "Missing artifact: " + name)
    return path


def verify_files(folder, entries):
    require(isinstance(entries, dict) and entries, "Missing SHA entries")
    for name, checksum in entries.items():
        require(isinstance(checksum, str) and re.fullmatch(r"[a-f0-9]{64}", checksum), "Invalid SHA entry")
        require(sha(artifact(folder, name)) == checksum, "Checksum mismatch: " + name)


def count(value):
    require(type(value) is int and value >= 0, "Invalid diagnostic count")
    return value


def fraction(numerator, denominator):
    numerator, denominator = count(numerator), count(denominator)
    require(numerator <= denominator, "Diagnostic numerator exceeds denominator")
    return f"{numerator} / {denominator}" + (" (not evaluated)" if not denominator else "")


@dataclass
class Export:
    policy: str
    summary: dict
    rows: list
    status: dict
    manifest_sha: str
    issues: list

    @property
    def metrics(self):
        return self.summary["mechanical_metrics"]

    @property
    def variants(self):
        return VARIANTS + ((OPTIONAL,) if OPTIONAL in self.metrics else ())


def load_export(folder, policy):
    """Check the producer's result manifest and the later collection snapshot."""
    folder = Path(folder).resolve()
    manifest = read_json(artifact(folder, "results-manifest.json"))
    collection = read_json(artifact(folder, "collection.json"))
    require(manifest["kind"] == collection["kind"] == KIND, "Wrong export kind")
    verify_files(folder, manifest["files"])
    verify_files(folder, collection["files"])
    require({"summary.json", "review.json"} <= manifest["files"].keys(), "Unbound summary or review")
    require("results-manifest.json" in collection["files"], "Unbound results manifest")
    require(all(collection["files"].get(name) == checksum for name, checksum in manifest["files"].items()),
            "Collection and results manifest disagree")
    summary = read_json(folder / "summary.json")
    review = read_json(folder / "review.json")
    binding = manifest["binding"]
    require(summary["kind"] == review["kind"] == binding["kind"] == KIND, "Wrong report kind")
    require(summary["binding"] == binding, "Summary binding differs")
    require(summary["generation_policy"] == binding["generation_policy"] == policy, "Policy mismatch")
    require(binding["guard_policy"] == ("repeat-block-v1" if policy == "guarded" else None), "Guard policy differs")
    require(all(collection["receipt"][key] == binding[key] for key in ("execution_sha256", "run_id")),
            "Collection receipt binding differs")
    metrics, rows, issues = summary["mechanical_metrics"], review["rows"], []
    require(isinstance(metrics, dict) and metrics and isinstance(rows, list), "Missing report data")
    require(set(metrics) <= set(VARIANTS + (OPTIONAL,)) and set(metrics) == set(binding["variants"])
            and len(metrics) == len(binding["variants"]), "Variant binding differs")
    for variant in VARIANTS:
        if variant not in metrics:
            issues.append(f"Missing required variant: {variant}")
    seen = set()
    for row in rows:
        require(row["kind"] == KIND and row["generation_policy"] == policy, "Review policy mismatch")
        require(row["variant"] in metrics and row["evaluation_type"] in {"fixed", "rollout"}, "Invalid review variant/type")
        require(isinstance(row["case_id"], str) and row["case_id"] and type(row["turn"]) is int
                and row["turn"] >= 0 and (row["evaluation_type"] != "fixed" or row["turn"] == 0), "Invalid case/turn")
        key = (row["evaluation_type"], row["case_id"], row["turn"], row["variant"])
        require(key not in seen, "Duplicate review slot")
        seen.add(key)
        require(type(row["verified"]) is bool and row["state"] in
                {"pending", "complete", "incomplete", "paused", "error", "blocked", "indeterminate"}, "Invalid review state")
        require(row["output"] is None or isinstance(row["output"], str), "Invalid output")
        require(row["complete"] is None or type(row["complete"]) is bool, "Invalid completion flag")
        require(row["checkpoint"] == metrics[row["variant"]]["identity"], "Checkpoint identity differs")
        if row["verified"]:
            require(isinstance(row["output"], str) and isinstance(row["messages"], list)
                    and isinstance(row["finish_reason"], str) and row["state"] in {"complete", "incomplete", "paused"},
                    "Verified reply lacks output/input/completion evidence")
            require(row["complete"] is (row["state"] == "complete")
                    and (row["state"] != "complete" or row["finish_reason"] == "eos")
                    and (row["state"] != "paused" or (policy == "guarded" and row["finish_reason"] == "cancelled")),
                    "Reply completion state differs")
        if row["state"] == "pending":
            require(not row["verified"] and row["record"] is None, "Pending row has result evidence")
        if row["record"] is not None:
            require(row["record"] in manifest["files"], "Unbound review record")
        require(not row["verified"] or row["record"] is not None, "Verified reply lacks record")
    expected = finalized = 0
    for variant, metric in metrics.items():
        chosen = [r for r in rows if r["variant"] == variant]
        verified = [r for r in chosen if r["verified"]]
        chat, pending = count(metric["expected_chat_slots"]), count(metric["pending_chat_slots"])
        require(chat > 0 and pending <= chat, "Invalid chat coverage")
        if len(chosen) != chat:
            issues.append(f"{variant}: {len(chosen)} / {chat} review rows present")
        require(len(chosen) <= chat, "Too many review rows")
        states = metric["chat_states"]
        require(sum(count(n) for n in states.values()) == chat - pending, "Chat state totals differ")
        require(count(metric["chat_verified"]) <= chat - pending, "Verified count exceeds finalized chat")
        if len(chosen) == chat:
            require(dict(Counter(r["state"] for r in chosen if r["state"] != "pending")) == states
                    and len(verified) == metric["chat_verified"]
                    and dict(Counter(r["finish_reason"] for r in verified)) == metric["finish_reasons"],
                    "Review and diagnostic counts differ")
        if pending:
            issues.append(f"{variant}: {pending} / {chat} chat slots pending")
        if metric["chat_verified"] < chat:
            issues.append(f"{variant}: {metric['chat_verified']} / {chat} chat replies verified")
        loss = metric["language_loss"]
        loss_expected, measured = count(loss["expected_rows"]), count(loss["measured_rows"])
        loss_finalized = sum(count(n) for n in loss["states"].values())
        require(measured <= loss_finalized <= loss_expected, "Invalid loss coverage")
        if measured < loss_expected:
            issues.append(f"{variant}: {measured} / {loss_expected} language-loss rows measured")
        expected += chat + loss_expected
        finalized += chat - pending + loss_finalized
        # Validate every displayed fraction now, even if the export is partial.
        diagnostic_rows(metric, chosen)
    require(count(summary["expected_slots"]) == expected
            and count(summary["finalized_slots"]) == finalized, "Summary slot totals differ")
    if finalized != expected:
        issues.append(f"{finalized} / {expected} total slots finalized")
    status = {}
    # status.json is intentionally outside results-manifest; collect() hashes it later.
    if "status.json" in collection["files"]:
        status = read_json(folder / "status.json")
        require(all(status[key] == value for key, value in binding.items()), "Status binding differs")
        if status.get("state") != "completed":
            issues.append("Run state: " + str(status.get("state", "unavailable")))
        for key in ("expected_slots", "finalized_slots"):
            if status.get(key) != summary[key]:
                issues.append(f"Status {key}: {status.get(key, 'unavailable')}; summary: {summary[key]}")
    else:
        issues.append("Completion state unavailable: status.json is not in collection.json")
    return Export(policy, summary, rows, status, sha(folder / "results-manifest.json"), issues)


def diagnostic_rows(metric, rows):
    chat, verified = metric["expected_chat_slots"], metric["chat_verified"]
    objective, loss = metric["strict_json"], metric["language_loss"]
    groups = Counter(r["case_id"] for r in rows if r["verified"] and r["evaluation_type"] == "rollout")
    pairs = sum(max(0, n - 1) for n in groups.values())
    identical = count(metric["consecutive_identical_rollout_replies"])
    # The producer compares successive verified replies, not necessarily adjacent turns.
    identical_value = (fraction(identical, pairs) if len(rows) == chat else
                       f"{identical} / unavailable (review rows missing)")
    result = [
        ("Finalized chat / planned chat", fraction(chat - metric["pending_chat_slots"], chat)),
        ("Verified replies / planned chat", fraction(verified, chat)),
        ("EOS replies / verified replies", fraction(metric["finish_reasons"].get("eos", 0), verified)),
        ("Guard-paused replies / verified replies", fraction(metric["guarded_pauses"], verified)),
        ("Repeated 4-gram replies / verified replies", fraction(metric["replies_with_repeated_4grams"], verified)),
        ("Identical successive verified rollout replies / compared pairs", identical_value),
        ("Strict JSON evaluated / expected cases", fraction(objective["evaluated"], objective["expected_cases"])),
        ("Strict JSON literal matches / evaluated cases", fraction(objective["literal_matches"], objective["evaluated"])),
        ("Strict JSON matches with EOS / evaluated cases", fraction(objective["literal_match_with_eos"], objective["evaluated"])),
        ("Language-loss measured / expected rows", fraction(loss["measured_rows"], loss["expected_rows"])),
    ]
    tokens, nll = count(loss["predicted_tokens"]), loss["summed_nll"]
    require(type(nll) in (int, float) and math.isfinite(nll) and nll >= 0, "Invalid summed NLL")
    mean = loss["token_weighted_mean_nll"]
    require((tokens == 0 and mean is None) or (tokens > 0 and type(mean) in (int, float)
            and math.isfinite(mean) and math.isclose(mean, nll / tokens)), "Invalid mean NLL")
    result.append(("Token-weighted NLL: summed NLL / predicted tokens", "Not measured" if not tokens else
                   f"{nll} / {tokens} = {mean}"))
    expected_ppl = math.exp(mean) if mean is not None and mean < 709 else None
    stored_ppl = loss["exp_mean_nll"]
    require((expected_ppl is None and stored_ppl is None) or
            (type(stored_ppl) in (int, float) and math.isfinite(stored_ppl)
             and expected_ppl is not None and math.isclose(stored_ppl, expected_ppl)),
            "Perplexity differs from token-weighted loss")
    result.append(("Held-out text perplexity: exp(mean NLL)",
                   "Not measured" if expected_ppl is None else f"{expected_ppl:.4f}"))
    result.append(("Chat states (counts)", "; ".join(f"{key}: {value}" for key, value in sorted(metric["chat_states"].items())) or "None"))
    return result


def compare_exports(raw, guarded):
    require(raw.policy == "raw" and guarded.policy == "guarded", "Separate raw and guarded exports required")
    for key in ("evaluation_sha256", "inference_sha256"):
        require(raw.summary["binding"][key] == guarded.summary["binding"][key], "Incomparable exports: " + key)
    for variant in set(raw.metrics) & set(guarded.metrics):
        require(raw.metrics[variant]["identity"] == guarded.metrics[variant]["identity"], "Cross-policy checkpoint differs")
    expected_groups = {(r["evaluation_type"], r["case_id"], r["turn"]) for e in (raw, guarded) for r in e.rows}
    for export in (raw, guarded):
        for variant in export.metrics:
            present = {(r["evaluation_type"], r["case_id"], r["turn"]) for r in export.rows if r["variant"] == variant}
            if present != expected_groups:
                issue = f"{variant}: {len(expected_groups - present)} case/turn groups missing from paired exports"
                if issue not in export.issues:
                    export.issues.append(issue)


def escaped(value):
    return html.escape(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def details(title, value):
    return f"<details><summary>{escaped(title)}</summary><pre><code>{escaped(value)}</code></pre></details>"


STYLE = """
:root{color-scheme:light;font:14px/1.5 system-ui,sans-serif;color:#222b29;background:#fff;letter-spacing:0}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:24px}
h1{font-size:25px;margin:0 0 8px}h2{font-size:20px;margin:24px 0 8px}h3{font-size:17px;margin:0 0 8px}
p{margin:6px 0 12px}.muted{color:#58615e}.notice{border-left:3px solid #b56518;padding-left:12px}
.filters{display:flex;flex-wrap:wrap;gap:16px;margin:20px 0;align-items:end}
label{display:grid;gap:4px}input,select{font:inherit;padding:7px;border:1px solid #99aaa4;border-radius:4px;background:#fff;max-width:100%}
input{width:300px}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;table-layout:fixed}
th,td{border:1px solid #cbd6d0;padding:10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:#f0f5f2;font-weight:600}caption{text-align:left;font-weight:600;margin-bottom:8px}
.diagnostics th:first-child{width:32%}.outputs{min-width:680px}.outputs th:first-child{width:150px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:normal;margin:8px 0;font-size:13px}
code{font:inherit}.output{font:14px/1.6 system-ui,sans-serif;tab-size:4}.state{font-size:12px;color:#58615e}
details{margin:8px 0}summary{cursor:pointer;color:#246253}.case-group{border-top:2px solid #cbd6d0;padding:18px 0}
.policy-outputs{margin:14px 0 22px}.turn-note{font-size:12px;font-weight:400;color:#58615e;display:block;margin-top:8px}
[hidden]{display:none!important}:focus-visible{outline:2px solid #187e69;outline-offset:3px}
@media(max-width:600px){main{padding:16px}h1{font-size:22px}.diagnostics{min-width:680px}.filters label{width:100%}input{width:100%}}
@media print{.filters{display:none}main{max-width:none;padding:0}.table-scroll{overflow:visible}.outputs{min-width:0}}
"""

SCRIPT = """
const query = document.getElementById('case-filter');
const policy = document.getElementById('policy-filter');
function filter() {
  const term = query.value.trim().toLowerCase();
  document.querySelectorAll('[data-policy]').forEach(section => {
    section.hidden = policy.value !== 'all' && section.dataset.policy !== policy.value;
  });
  let visible = 0;
  document.querySelectorAll('.case-group').forEach(section => {
    section.hidden = !section.dataset.search.toLowerCase().includes(term) ||
      !Array.from(section.querySelectorAll('[data-policy]')).some(part => !part.hidden);
    if (!section.hidden) visible++;
  });
  document.getElementById('case-count').textContent = visible + ' case groups';
  document.getElementById('empty').hidden = visible !== 0;
}
query.addEventListener('input', filter);
policy.addEventListener('change', filter);
filter();
"""


def output_cell(row):
    if row is None:
        return '<td class="missing">Missing variant or case/turn row; no output available.</td>'
    state = {key: row[key] for key in ("state", "verified", "finish_reason", "complete")}
    text = ('<p class="muted">No output recorded.</p>' if row["output"] is None else
            '<pre class="output"><code>' + escaped(row["output"]) + '</code></pre>')
    if row["output"] == "":
        text += '<p class="muted">Recorded output is an empty string.</p>'
    return '<td><p class="state">' + escaped("; ".join(f"{k}: {v}" for k, v in state.items())) + '</p>' + text + \
        details("Exact inputs", row["messages"]) + details("Rubric", row["review_criteria"]) + \
        details("Checkpoint and human review", {k: row[k] for k in ("checkpoint", "human_review")}) + '</td>'


def render_report(raw, guarded, *, allow_partial=False):
    compare_exports(raw, guarded)
    exports = (raw, guarded)
    issues = [f"{e.policy}: {issue}" for e in exports for issue in e.issues]
    require(allow_partial or not issues, "Partial exports; use --allow-partial to label them explicitly: " + "; ".join(issues))
    script_hash = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
             f"script-src 'sha256-{script_hash}'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">",
             '<title>R4 evaluation comparison</title><style>' + STYLE + '</style></head><body><main>',
             '<h1>R4 evaluation comparison</h1><p class="muted">Private development evaluation, not blind final. '
             'Mechanical diagnostics only; no semantic, native-human or clinical certification.</p>',
             '<p><strong>' + ('PARTIAL comparison' if issues else 'Completed runs; full reported slot coverage') + '</strong>. '
             'Run completion does not mean every reply completed with EOS.</p>']
    if issues:
        parts.append('<ul class="notice">' + ''.join('<li>' + escaped(i) + '</li>' for i in issues) + '</ul>')
    parts.append('<p class="muted">All results-manifest and collection SHA entries checked locally. '
                 'These checks establish snapshot consistency, not independent authenticity or model quality.</p>')
    parts.append('<div class="filters"><label>Case or cohort<input id="case-filter" type="search"></label>'
                 '<label>Policy<select id="policy-filter"><option value="all">Raw and guarded</option>'
                 '<option value="raw">Raw only</option><option value="guarded">Guarded only</option></select></label>'
                 '<span id="case-count" role="status" aria-live="polite"></span></div>')
    parts.append('<h2>Whole-export diagnostics</h2><p class="muted">Counts remain whole-export when case search is active. '
                 'Raw and guarded are separate policies; no pooled score or improvement claim. '
                 'Zero denominators are not evaluated. R4 pass1 is optional in raw.</p>')
    for export in exports:
        parts.append('<section data-policy="' + export.policy + '"><h3>' + export.policy.capitalize() + '</h3>')
        parts.append('<p>' + escaped(f"Run state: {export.status.get('state', 'unavailable')}; "
                     f"finalized slots: {export.summary['finalized_slots']} / {export.summary['expected_slots']}") + '</p>')
        tables = {v: diagnostic_rows(m, [r for r in export.rows if r["variant"] == v]) for v, m in export.metrics.items()}
        names = next(iter(tables.values()))
        parts.append('<div class="table-scroll"><table class="diagnostics"><thead><tr><th scope="col">Diagnostic</th>' +
                     ''.join('<th scope="col">' + LABELS[v] + '</th>' for v in export.variants) + '</tr></thead><tbody>')
        for index, (name, _) in enumerate(names):
            parts.append('<tr><th scope="row">' + escaped(name) + '</th>' + ''.join(
                '<td>' + escaped(tables[v][index][1] if v in tables else 'Missing variant; not measured') + '</td>'
                for v in export.variants) + '</tr>')
        parts.append('</tbody></table></div>' + details('Binding and review status', {
            'results_manifest_sha256': export.manifest_sha, 'binding': export.summary['binding'],
            'human_quality': export.summary['human_quality']}) + '</section>')
    grouped = defaultdict(dict)
    cohorts = defaultdict(set)
    for export in exports:
        for row in export.rows:
            group = (row["evaluation_type"], row["case_id"])
            grouped[group][(export.policy, row["turn"], row["variant"])] = row
            if row["cohort"]:
                cohorts[group].add(str(row["cohort"]))
    parts.append('<h2>Exact outputs</h2><p id="empty" hidden>No matching cases.</p>')
    for group, records in sorted(grouped.items()):
        kind, case = group
        parts.append('<article class="case-group" data-case="' + escaped(case) + '" data-search="' +
                     escaped(' '.join([case, *sorted(cohorts[group])])) + '"><h3>' + escaped(case) +
                     ' <span class="muted">(' + kind + ')</span></h3>')
        turns = sorted({key[1] for key in records})
        for export in exports:
            parts.append('<section class="policy-outputs" data-policy="' + export.policy + '">'
                         '<div class="table-scroll"><table class="outputs"><caption>' + export.policy.capitalize() +
                         '</caption><thead><tr><th scope="col">Input / turn</th>' + ''.join(
                             '<th scope="col">' + LABELS[v] + '</th>' for v in export.variants) + '</tr></thead><tbody>')
            for turn in turns:
                label = 'Fixed input' if kind == 'fixed' else f'Turn {turn + 1}'
                note = ('Own-model history: prior replies belong to this variant and policy; inputs are not a shared prompt.'
                        if kind == 'rollout' and turn > 0 else 'Fixed supplied history.' if kind == 'fixed' else 'First rollout input.')
                prompts = {row["messages"][-1]["content"] for variant in export.variants
                           if (row := records.get((export.policy, turn, variant))) and row["messages"]}
                prompt_text = (next(iter(prompts)) if len(prompts) == 1 else
                               "No verified input available." if not prompts else "Inputs differ; inspect each model's exact inputs.")
                parts.append('<tr data-turn="' + str(turn) + '"><th scope="row">' + label +
                             '<pre class="prompt">' + escaped(prompt_text) + '</pre>' +
                             '<span class="turn-note">' + note + '</span></th>' + ''.join(
                                 output_cell(records.get((export.policy, turn, variant))) for variant in export.variants) + '</tr>')
            parts.append('</tbody></table></div></section>')
        parts.append('</article>')
    parts.append('</main><script>' + SCRIPT + '</script></body></html>')
    return '\n'.join(parts)


def write_private(path, document):
    path = Path(path).absolute()
    private = ROOT / "tmp"
    require(path.suffix.lower() == ".html" and path.resolve().is_relative_to(private)
            and not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(ROOT)),
            "Output must be an unlinked .html path under repository tmp/")
    ignored = subprocess.run(["git", "check-ignore", "--quiet", "--", str(path)], cwd=ROOT, check=False)
    require(ignored.returncode == 0, "Output must be git-ignored")
    require(not path.exists(), "Output already exists; choose a fresh private report path")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(document)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="Collected raw export directory")
    parser.add_argument("--guarded", required=True, type=Path, help="Collected guarded export directory")
    parser.add_argument("--out", required=True, type=Path, help="Fresh .html path under repository tmp/")
    parser.add_argument("--allow-partial", action="store_true", help="Label missing coverage instead of rejecting it")
    args = parser.parse_args(argv)
    try:
        raw, guarded = load_export(args.raw, "raw"), load_export(args.guarded, "guarded")
        document = render_report(raw, guarded, allow_partial=args.allow_partial)
        path = write_private(args.out, document)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("Report not written: " + str(error), file=sys.stderr)
        return 1
    print("Private report: " + str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
