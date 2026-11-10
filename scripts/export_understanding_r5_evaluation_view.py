#!/usr/bin/env python3
"""Private offline comparison: --results LABEL=DIR (repeatable), --agent-review FILE,
--out NEW_DIRECTORY. Reads hash-bound runner reports; never generates or scores text.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import run_understanding_r5_evaluation as r

VARIANTS = {"base": "Base", "r4_response": "R4", "r5_response": "R5"}
STATES = ("complete", "incomplete", "paused", "error", "blocked", "indeterminate", "pending")


def unchanged(hashes):
    r.require(all(not p.is_symlink() and r.sha(p) == h for p, h in hashes.items()), "Source changed during export")


def check_counts(rows, counts):
    states = Counter(row["state"] for row in rows)
    groups = {row["source_group_id"] for row in rows if row.get("source_group_id")}
    expected = {"expected_slots": len(rows), "expected_cases": len({row["case_id"] for row in rows}),
                "source_group_count": len(groups), "source_group_ids": sorted(groups),
                "unclassified_source_group_cases": len({row["case_id"] for row in rows if not row.get("source_group_id")}),
                "finalized_slots": sum(row["record"] is not None for row in rows),
                "verified_slots": sum(row["verified"] for row in rows),
                "states": {s: states[s] for s in STATES}}
    r.require(all(counts[k] == v for k, v in expected.items()) and counts["meaning_accuracy"] is None,
              "Mechanical denominators differ from retained rows")
    literals = [row["strict_literal"] for row in rows if row["verified"] and row["strict_literal"] is not None]
    literal = counts["strict_literal"]
    r.require(literal["evaluated_slots"] == len(literals)
              and literal["literal_matches"] == sum(v["literal_match"] for v in literals)
              and literal["literal_matches_with_eos"] == sum(v["passed"] for v in literals)
              and len(literals) <= literal["expected_slots"] <= len(rows), "Literal denominators differ")


def load_results(label, folder):
    folder = Path(folder).resolve(strict=True)
    manifest_path = r.safe(folder, "results-manifest.json")
    hashes = {manifest_path: r.sha(manifest_path)}
    manifest = r.read(manifest_path)
    binding = manifest["binding"]
    r.require(binding["variants"] == list(VARIANTS) and binding["generation_policy"] in {"raw", "guarded"}
              and binding["kind"] in {r.KIND, r.FRESH_KIND}, "Expected separate base/R4/R5 R5 suite/policy")
    for name, checksum in manifest["files"].items():
        path = r.safe(folder, name)
        r.require(r.sha(path) == checksum, "Result artifact changed: " + name)
        hashes[path] = checksum
    optional = {"results-manifest.json", "collection.json", "DERIVED-PARTIAL-REPORT.json", "status.json"}
    r.require(set(r.files(folder)) <= set(manifest["files"]) | optional, "Unexpected unbound result file")
    required = {"execution.json", "journal/binding.json", "summary.json", "benchmark.json", "review.json", "language-loss-cases.json"}
    r.require(required <= set(manifest["files"]), "Missing bound report evidence")
    for name in optional - {"results-manifest.json"}:
        path = r.safe(folder, name)
        if path.exists():
            hashes[path] = r.sha(path)
    collection = None
    if (folder / "collection.json").exists():
        collection = r.read(folder / "collection.json")
        r.require(all(collection["receipt"][k] == binding[k] for k in ("run_id", "execution_sha256")), "Collection binding differs")
        for name, checksum in collection["files"].items():
            path = r.safe(folder, name)
            r.require(r.sha(path) == checksum, "Collected artifact changed: " + name)
            hashes[path] = checksum
    derived = r.read(folder / "DERIVED-PARTIAL-REPORT.json") if (folder / "DERIVED-PARTIAL-REPORT.json").exists() else None
    if derived:
        r.require(collection is not None and derived["collection_sha256"] == hashes[folder / "collection.json"]
                  and derived["execution_sha256"] == binding["execution_sha256"] and derived["generation_requests"] == 0,
                  "Partial report derivation differs")
    execution, summary, benchmark, review, loss = (r.read(folder / n) for n in
        ("execution.json", "summary.json", "benchmark.json", "review.json", "language-loss-cases.json"))
    r.require(hashes[folder / "execution.json"] == binding["execution_sha256"]
              and summary["binding"] == benchmark["binding"] == r.read(folder / "journal/binding.json") == binding
              and all(v["kind"] == binding["kind"] for v in (manifest, summary, review, loss, execution))
              and execution["generation_policy"] == summary["generation_policy"] == benchmark["generation_policy"] == binding["generation_policy"]
              and summary["benchmark"] == {k: v for k, v in benchmark.items() if k != "binding"}, "Report identity differs")
    records = {}
    for name in manifest["files"]:
        if not name.startswith("journal/records/"):
            continue
        envelope = r.read(folder / name)
        record = envelope["record"]
        r.require(envelope["sha256"] == r.digest(record) and record["binding_sha256"] == r.digest(binding)
                  and name == "journal/records/" + r.digest(record["id"]) + ".json" and record["id"] not in records
                  and record["inputs_sha256"] == r.digest(record["request"]), "Raw record hash/binding differs")
        proof_name = "runtime/attempt-" + str(record["attempt"]) + ".json"
        r.require(proof_name in manifest["files"] and manifest["files"][proof_name] == record["runtime_proof_sha256"]
                  and r.read(folder / proof_name)["binding"] == binding, "Raw runtime proof differs")
        records[record["id"]] = record
    rows, seen = review["rows"], set()
    for row in rows:
        key, record = row["slot_id"], records.get(row["slot_id"])
        r.require(key not in seen and row["variant"] in VARIANTS and row["state"] in STATES
                  and row["kind"] == binding["kind"] and row["generation_policy"] == binding["generation_policy"]
                  and type(row["turn"]) is int and row["turn"] >= 0, "Invalid/duplicate review slot")
        expected_key = row["evaluation_type"] + ":" + row["case_id"] + ":" + row["variant"]
        r.require(row["evaluation_type"] in {"fixed", "rollout"}
                  and key == expected_key + (":" + str(row["turn"]) if row["evaluation_type"] == "rollout" else "")
                  and (row["evaluation_type"] != "fixed" or row["turn"] == 0), "Noncanonical review slot")
        seen.add(key)
        raw = (record or {}).get("result", {})
        expected = {"state": record["state"] if record else "pending", "verified": bool(record and record.get("verified")),
                    "record": "journal/records/" + r.digest(key) + ".json" if record else None,
                    "messages": (record or {}).get("request", {}).get("messages"),
                    "strict_literal": (record or {}).get("objective"),
                    **{k: raw.get(k) for k in ("output", "finish_reason", "complete")},
                    **{k: (record or {}).get(k) for k in ("error", "error_type", "reason")}}
        r.require(all(row.get(k) == value for k, value in expected.items()), "Review output/status differs from raw record")
        if record:
            r.require(all(row[k] == record[k] for k in ("case_id", "variant", "turn"))
                      and row["evaluation_type"] == record["kind"] and record["policy"] == row["generation_policy"], "Raw slot identity differs")
            r.require(row["checkpoint"] == summary["mechanical_metrics"][row["variant"]]["identity"], "Checkpoint identity differs")
    loss_ids = set()
    for row in loss["rows"]:
        key, record = row["slot_id"], records.get(row["slot_id"])
        r.require(key not in seen | loss_ids and row["variant"] in VARIANTS
                  and row["state"] == (record["state"] if record else "pending"), "Invalid retention row")
        loss_ids.add(key)
        if record:
            r.require(record["kind"] == "loss" and row["case_id"] == record["case_id"] and row["variant"] == record["variant"]
                      and row["measured"] == (record.get("verified") is True and record["state"] == "measured"), "Retention identity differs")
            if row["measured"]:
                r.require(row["measured_predicted_tokens"] == record["loss"]["predicted_tokens"]
                          and row["summed_nll"] == record["loss"]["summed_nll"], "Retention measurement differs")
        if not record or not row["measured"]:
            r.require(row["measured"] is False and row["measured_predicted_tokens"] == 0 and row["summed_nll"] is None,
                      "Unmeasured retention row cannot carry a score")
    r.require(set(records) <= seen | loss_ids and summary["finalized_slots"] == len(records)
              and summary["expected_slots"] == execution["expected_slots"] == len(rows) + len(loss_ids), "Coverage omits/duplicates records")
    if derived:
        r.require(derived["saved_records"] == len(records) and derived["planned_slots"] == summary["expected_slots"], "Partial coverage differs")
    for variant in VARIANTS:
        chosen = [row for row in rows if row["variant"] == variant]
        check_counts(chosen, benchmark["totals"][variant])
        metric = summary["mechanical_metrics"][variant]
        r.require(metric["expected_chat_slots"] == len(chosen)
                  and metric["pending_chat_slots"] == sum(row["record"] is None for row in chosen)
                  and metric["chat_states"] == dict(Counter(row["state"] for row in chosen if row["record"])), "Chat coverage differs")
        r.require(set(metric["language_loss"]) == {row["cohort"] for row in loss["rows"] if row["variant"] == variant},
                  "Retention cohort omitted")
        for cohort, counts in metric["language_loss"].items():
            selected = [row for row in loss["rows"] if row["variant"] == variant and row["cohort"] == cohort]
            tokens = sum(row["measured_predicted_tokens"] for row in selected)
            measured = [row for row in selected if row["measured"]]
            nll = sum(row["summed_nll"] for row in measured) if measured else None
            states = Counter(row["state"] for row in selected)
            known = [row["expected_predicted_tokens"] for row in selected if row["expected_predicted_tokens"] is not None]
            r.require(counts["expected_rows"] == len(selected) and counts["states"] == dict(Counter(row["state"] for row in selected))
                      and counts["measured_rows"] == len(measured) and counts["predicted_tokens"] == tokens
                      and counts["pending_rows"] == states["pending"] and counts["error_rows"] == states["error"]
                      and counts["indeterminate_rows"] == states["indeterminate"]
                      and counts["expected_predicted_tokens"] == (sum(known) if len(known) == len(selected) else None)
                      and counts["summed_nll"] == nll and counts["token_weighted_mean_nll"] == (nll / tokens if tokens else None)
                      and counts["complete_coverage"] == (bool(selected) and len(measured) == len(selected)), "Retention denominators differ")
    for group in benchmark["groups"]:
        for variant in VARIANTS:
            selected = [row for row in rows if row["variant"] == variant and all(
                value in row["benchmark_labels"]["categories"] if key == "category" else row["benchmark_labels"][key] == value
                for key, value in group["labels"].items())]
            check_counts(selected, group["models"][variant])
    started = set()
    for name in manifest["files"]:
        if name.startswith("journal/attempts/"):
            marker = r.read(folder / name)
            r.require(marker["state"] == "started" and Path(name).stem == r.digest(marker["id"]), "Invalid start marker")
            started.add(marker["id"])
    unchanged(hashes)
    return {"label": label, "binding": binding, "summary": summary, "benchmark": benchmark, "rows": rows,
            "raw_chat": {k: v for k, v in records.items() if k in seen}, "loss_cases": loss["rows"],
            "started_without_result": len(started - records.keys()), "agent": {},
            "proof": {"directory": str(folder), "results_manifest_sha256": hashes[manifest_path],
                      "review_sha256": hashes[folder / "review.json"], "collection": collection, "derived": derived,
                      "files": {str(p.relative_to(folder)): h for p, h in hashes.items()}}}, hashes


def attach_reviews(results, paths):
    hashes, sources = {}, []
    for path in paths:
        path = Path(path).absolute()
        r.require(not path.is_symlink() and path not in hashes, "Duplicate/linked agent review")
        hashes[path] = r.sha(path)
        value = r.read(path)
        source_hashes = [value[k] for k in ("source_file_sha256", "source_sha256") if k in value]
        matches = [result for result in results if source_hashes and all(h == result["proof"]["review_sha256"] for h in source_hashes)]
        r.require(value["version"] == "r5-agent-semantic-review-v1"
                  and value["reviewer_type"] == "agent_not_native_human_or_clinical" and len(matches) == 1,
                  "Agent review must bind exactly one supplied review.json")
        result = matches[0]
        index = {row["slot_id"]: row for row in result["rows"]}
        source = {"path": str(path), "sha256": hashes[path], "metadata": {k: v for k, v in value.items() if k not in {"rows", "summary"}}}
        sources.append(source)
        for rating in value["rows"]:
            key = rating["slot_id"]
            r.require(key in index and key not in result["agent"] and index[key]["record"] is not None
                      and all(rating[k] == index[key][k] for k in ("case_id", "variant", "turn")), "Duplicate/unbound agent judgment")
            r.require(all(rating[k] in {"pass", "partial", "fail", "uncertain", "not_applicable", "pending", None}
                          for k in ("meaning", "usefulness", "requested_language"))
                      and type(rating["unsupported_claims"]) in (bool, type(None)) and isinstance(rating["notes"], str), "Invalid agent judgment")
            result["agent"][key] = {"rating": rating, "source": source}
    return sources, hashes


def escaped_json(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character in "<>&\u2028\u2029":
        text = text.replace(character, "\\u%04x" % ord(character))
    return text


def attach_phrase_reviews(results, paths):
    hashes, sources, events = {}, [], set()
    for path in paths:
        path = Path(path).absolute()
        r.require(not path.is_symlink() and path not in hashes, "Duplicate/linked phrase review")
        hashes[path] = r.sha(path)
        value = r.read(path)
        r.require(value["version"] == "r5-user-phrase-review-v1" and value["reviewer_type"] == "user"
                  and value["review_scope"] == "quoted_phrase_only" and value["full_response_reviewed"] is False
                  and value["training_eligible"] is False and value["original_output_modified"] is False,
                  "Phrase review must not imply full-response approval or training admission")
        r.require(isinstance(value["event_id"], str) and value["event_id"] and value["event_id"] not in events
                  and all(isinstance(value[k], str) and value[k].strip() for k in
                          ("question", "user_answer", "original_quote", "corrected_quote"))
                  and value["original_quote"] != value["corrected_quote"], "Invalid phrase review evidence")
        events.add(value["event_id"])
        source = {"path": str(path), "sha256": hashes[path], "review": value}
        sources.append(source)
        r.require(isinstance(value["occurrences"], list) and value["occurrences"], "Missing phrase occurrences")
        seen = set()
        for occurrence in value["occurrences"]:
            matches = [d for d in results if d["proof"]["review_sha256"] == occurrence["source_file_sha256"]]
            key = (occurrence["source_file_sha256"], occurrence["slot_id"])
            r.require(len(matches) == 1 and key not in seen, "Unbound/duplicate phrase occurrence")
            seen.add(key)
            result = matches[0]
            rows = [row for row in result["rows"] if row["slot_id"] == occurrence["slot_id"]]
            r.require(len(rows) == 1 and all(rows[0][k] == occurrence[k] for k in ("case_id", "variant", "turn"))
                      and isinstance(rows[0]["output"], str) and value["original_quote"] in rows[0]["output"],
                      "Phrase review does not match saved output")
            result.setdefault("phrase_reviews", {}).setdefault(occurrence["slot_id"], []).append(source)
    return sources, hashes


STYLE = """
*{box-sizing:border-box;letter-spacing:0}body{margin:0;background:#fff;color:#202b27;font:14px/1.5 system-ui,sans-serif}header{padding:20px 24px 12px;border-top:5px solid #24715d}h1{font-size:25px;margin:0}h2{font-size:18px;margin:0 0 10px}h3{font-size:15px;margin:16px 0 8px}p{margin:5px 0 12px}.muted{color:#5b6862}
.toolbar{display:flex;flex-wrap:wrap;gap:12px;padding:12px 24px;background:#f6f8f6;border-block:1px solid #dce4df}.toolbar label{display:grid;gap:4px;min-width:120px}.grow{flex:1}select,input,button{font:inherit;border:1px solid #b8c6bd;border-radius:4px;background:white;color:inherit;padding:8px;min-width:0;width:100%}select{max-width:100%}:focus-visible{outline:3px solid #41846b;outline-offset:2px}
section,footer{padding:16px 24px}.coverage{display:flex;flex-wrap:wrap;gap:12px 28px;padding-top:12px}.coverage strong{font-size:20px}.coverage span{display:block}.notice{color:#7b5513;background:#fff6df;padding:8px 12px;border-left:3px solid #bb902e}
.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce4df;padding:7px 9px}th{background:#f5f8f6;color:#4c5f54;white-space:nowrap}tbody th{background:white}caption{text-align:left;margin:8px 0;font-weight:600}
.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-block:1px solid #dce4df}.model{padding:16px;min-width:0;border-right:1px solid #dce4df}.model:last-child{border:0}.model h3{margin:0 0 6px;font-size:18px}.model:nth-child(2){border-top:3px solid #8c9fb7}.model:nth-child(3){border-top:3px solid #4b8f71}.model:first-child{border-top:3px solid #a7b3ac}
.output,.message,pre{white-space:pre-wrap;overflow-wrap:anywhere}.output{margin:12px 0;padding:10px 0;border-block:1px solid #e3e9e5}.message{margin:6px 0;padding:8px 10px;border-left:3px solid #cddbd1;background:#f8faf8}.message b{font-size:12px;color:#566a5e}.badges{display:flex;flex-wrap:wrap;gap:6px}.badge{font-size:12px;border-radius:3px;padding:3px 7px;background:#edf0ee}.pass{background:#e6f3e9;color:#285d38}.fail{background:#f8e9e8;color:#8a3330}.partial,.uncertain{background:#fff3d9;color:#785514}
summary{cursor:pointer;padding:10px 0;color:#285f4b}details{border-top:1px solid #e0e7e2}pre{font-size:12px;margin:4px 0}footer{background:#f6f8f6;border-top:1px solid #dce4df}.small{font-size:12px}.case-title{overflow-wrap:anywhere}.metric-select{max-width:500px;margin:8px 0}
@media(max-width:900px){.comparison{grid-template-columns:minmax(0,1fr)}.model{border-right:0;border-bottom:1px solid #dce4df}.toolbar label{width:100%;flex:auto}header,section,footer{padding:16px}.toolbar{padding:12px 16px}.coverage{gap:10px 18px}.coverage strong{font-size:18px}}
"""

SCRIPT = r"""
'use strict';
const data=JSON.parse(document.getElementById('snapshot').textContent),$=id=>document.getElementById(id),names={base:'Base',r4_response:'R4',r5_response:'R5'};
const node=(tag,text,parent)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(parent)parent.append(e);return e;};
const human=s=>String(s).replaceAll('_',' '),fmt=v=>v===null||v===undefined?'Not measured':typeof v==='number'?v.toLocaleString('en-US',{maximumFractionDigits:4}):String(v);
const details=(parent,title,value)=>{const d=node('details',undefined,parent);node('summary',title,d);node('pre',JSON.stringify(value,null,2),d);return d;};
function table(parent,headers,rows){const t=node('table',undefined,parent),head=node('tr',undefined,node('thead',undefined,t));for(const h of headers)node('th',h,head);const body=node('tbody',undefined,t);for(const row of rows){const tr=node('tr',undefined,body);row.forEach((v,i)=>node(i?'td':'th',fmt(v),tr));}return t;}
let result=data.results[0],groupKey='',turn=0,groups=[];
function groupRows(){const map=new Map();for(const row of result.rows){const key=JSON.stringify([row.evaluation_type,row.case_id]);if(!map.has(key))map.set(key,{key,id:row.case_id,type:row.evaluation_type,rows:[]});map.get(key).rows.push(row);}return [...map.values()];}
function metrics(){const benchmark=result.benchmark,idx=$('breakdown').value,g=idx===''?null:benchmark.groups[Number(idx)],totals=g?g.models:benchmark.totals;$('mechanical').replaceChildren();
 table($('mechanical'),['Model','Cases','Source groups','Cases without group','Slots','Saved','Complete','Incomplete','Error','Indeterminate','Blocked','Paused','Pending','Literal + EOS / evaluated / expected'],Object.entries(names).map(([v,n])=>{const c=totals[v],s=c.states;return[n,c.expected_cases,c.source_group_count,c.unclassified_source_group_cases,c.expected_slots,c.finalized_slots,s.complete,s.incomplete,s.error,s.indeterminate,s.blocked,s.paused,s.pending,`${c.strict_literal.literal_matches_with_eos} / ${c.strict_literal.evaluated_slots} / ${c.strict_literal.expected_slots}`];}));
 $('retention').replaceChildren();const rows=[];for(const [v,n] of Object.entries(names))for(const [cohort,m] of Object.entries(result.summary.mechanical_metrics[v].language_loss)){rows.push([n,human(cohort),m.measured_rows+' / '+m.expected_rows,m.pending_rows,m.error_rows,m.indeterminate_rows,m.predicted_tokens+' / '+fmt(m.expected_predicted_tokens),fmt(m.token_weighted_mean_nll),m.complete_coverage?'Complete':'Partial']);}
 if(rows.length)table($('retention'),['Model','Retention cohort','Measured / expected','Pending','Errors','Indeterminate','Tokens / expected','Mean NLL','Coverage'],rows);else node('p','Retention not included in this execution.',$('retention'));
}
function suite(){result=data.results[Number($('suite').value)];const s=result.summary,chat=result.rows.filter(r=>r.record).length,loss=result.loss_cases.filter(r=>r.measured).length;
 $('coverage').replaceChildren();for(const [label,value] of [['Finalized slots',`${s.finalized_slots} / ${s.expected_slots}`],['Chat slots saved',`${chat} / ${result.rows.length}`],['Retention measured',`${loss} / ${result.loss_cases.length}`],['Agent review entries',`${Object.keys(result.agent).length} / ${result.rows.length}`]]){const d=node('div',undefined,$('coverage'));node('strong',value,d);node('span',label,d);}
 $('scope').textContent=`${human(result.binding.kind)} / ${result.binding.generation_policy} / ${result.binding.run_id}`;
 $('partial').textContent=(s.finalized_slots<s.expected_slots?'Partial coverage. ':'All planned slots finalized; errors and incomplete outputs remain counted. ')+`${result.started_without_result} started request(s) without a saved result. Completion is not accuracy.`;
 $('breakdown').replaceChildren();node('option','All chat',$('breakdown')).value='';result.benchmark.groups.forEach((g,i)=>{node('option',human(g.dimension)+': '+Object.values(g.labels).join(' / '),$('breakdown')).value=i;});
 groupKey='';groups=groupRows();filter();metrics();$('proof').replaceChildren();details($('proof'),'Report and collection hash references',result.proof);details($('proof'),'Checkpoint identities',Object.fromEntries(Object.entries(result.summary.mechanical_metrics).map(([k,v])=>[k,v.identity])));
}
function filter(){const q=$('search').value.toLocaleLowerCase().trim(),matched=groups.filter(g=>JSON.stringify(g).toLocaleLowerCase().includes(q));$('case').replaceChildren();for(const g of matched)node('option',`${g.id} (${g.type})`,$('case')).value=g.key;
 if(!matched.some(g=>g.key===groupKey))groupKey=matched[0]?.key||'';$('case').value=groupKey;$('matches').textContent=`${matched.length} / ${groups.length} cases`;$('case').disabled=!matched.length;turns();}
function turns(){const g=groups.find(g=>g.key===groupKey),values=[...new Set((g?.rows||[]).map(r=>r.turn))].sort((a,b)=>a-b);$('turn').replaceChildren();for(const n of values)node('option','Turn '+(n+1),$('turn')).value=n;if(!values.includes(turn))turn=values[0]||0;$('turn').value=turn;$('turn').disabled=!values.length;compare();}
function compare(){const g=groups.find(g=>g.key===groupKey);$('comparison').replaceChildren();$('case-title').textContent=g?g.id:'No matching cases';if(!g)return;
 for(const [variant,name] of Object.entries(names)){const pane=node('article',undefined,$('comparison'));pane.className='model';node('h3',name,pane);const row=g.rows.find(r=>r.variant===variant&&r.turn===turn);if(!row){node('p','Slot not present in this report.',pane);continue;}
  node('p',`${human(row.state)} / stop: ${row.finish_reason||'Not recorded'}`,pane).className='muted';const rating=result.agent[row.slot_id];node('p',rating?.rating.transferred_judgment?'Agent judgment reused from an exact raw match':'Agent review',pane).className='small';const badges=node('div',undefined,pane);badges.className='badges';
  for(const [key,label] of [['meaning','Meaning'],['usefulness','Usefulness'],['requested_language','Language']]){const v=rating?.rating[key]||'pending';node('span',label+': '+human(v),badges).className='badge '+v;}
  if(row.messages?.length){const latest=[...row.messages].reverse().find(m=>m.role==='user');if(latest){node('h4','Latest user message',pane);node('div',latest.content,pane).className='message';}}
  else node('p','Exact prompt/history not recorded for this slot.',pane).className='muted';
  node('h4','Exact output',pane);node('div',row.output===null?'No saved output.':row.output===''?'Empty output (0 characters).':row.output,pane).className='output';
  if(row.error||row.reason)node('p',[row.error_type,row.error,row.reason].filter(Boolean).join(': '),pane).className='notice';
  if(rating){node('p',rating.rating.notes,pane);node('p','Agent unsupported-claim flag: '+(rating.rating.unsupported_claims===null?'uncertain':rating.rating.unsupported_claims?'flagged':'none identified'),pane).className='small';details(pane,'Agent review source',rating.source);if(rating.rating.transferred_judgment)details(pane,'Exact-match transfer evidence (not a new independent review)',rating.rating.transfer_proof);}
  for(const evidence of result.phrase_reviews?.[row.slot_id]||[]){node('h4','User-confirmed wording correction',pane);node('p','Phrase only; full-answer ratings and original output remain unchanged.',pane).className='small';node('p','Original: '+evidence.review.original_quote,pane);node('p','Correction: '+evidence.review.corrected_quote,pane);details(pane,'User confirmation and exact source',evidence);}
  const history=node('details',undefined,pane);node('summary',row.evaluation_type==='rollout'?'Exact prompt / own rollout history':'Exact prompt / supplied history',history);for(const m of row.messages||[]){const d=node('div',undefined,history);d.className='message';node('b',m.role,d);node('div',m.content,d);}
  details(pane,'Authored criteria and capability metadata',{criteria:row.review_criteria,labels:row.benchmark_labels,source_group_id:row.source_group_id,source_case_sha256:row.source_case_sha256});
  details(pane,'Raw record and hash reference',{source_directory:result.proof.directory,review_sha256:result.proof.review_sha256,record_path:row.record,record_file_sha256:row.record?result.proof.files[row.record]:null,review_row:row,raw_record:result.raw_chat[row.slot_id]||null});
 }
}
data.results.forEach((d,i)=>{node('option',`${d.label} / ${human(d.binding.kind)} / ${d.binding.generation_policy}`,$('suite')).value=i;});
$('suite').onchange=suite;$('search').oninput=filter;$('case').onchange=()=>{groupKey=$('case').value;turn=0;turns();};$('turn').onchange=()=>{turn=Number($('turn').value);compare();};$('breakdown').onchange=metrics;suite();
"""


def render_html(data):
    checksum = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'sha256-{checksum}'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>R5 evaluation comparison</title><style>{STYLE}</style></head><body><header><h1>R5 evaluation comparison</h1><p id="scope" class="muted"></p><div id="coverage" class="coverage"></div></header>
<div class="toolbar"><label class="grow">Suite / policy<select id="suite"></select></label></div><section><p id="partial" class="notice"></p><h2>Mechanical denominators</h2><label>Capability group<select id="breakdown" class="metric-select"></select></label><div id="mechanical" class="scroll"></div><p class="small muted">Source groups are unique within each model and subgroup. Category memberships overlap. Literal checks and completion are not meaning accuracy.</p><details><summary>English / Twi retention NLL</summary><div id="retention" class="scroll"></div></details></section>
<div class="toolbar"><label class="grow">Search cases and saved text<input id="search" type="search" autocomplete="off"></label><label class="grow">Case<select id="case"></select></label><label>Turn<select id="turn"></select></label></div>
<section><p id="matches" class="muted"></p><h2 id="case-title" class="case-title"></h2><div id="comparison" class="comparison"></div></section>
<footer><p>Private research. Agent judgments are not native-human or clinical certification. Unreviewed judgments remain pending. No aggregate meaning accuracy.</p><details><summary>Source verification</summary><div id="proof"></div></details></footer>
<noscript>JavaScript is required to display the embedded results.</noscript><script id="snapshot" type="application/json">{escaped_json(data)}</script><script>{SCRIPT}</script></body></html>"""


def export(results, agent_reviews, out, phrase_reviews=()):
    out = Path(out).absolute()
    r.require(results and not out.exists() and not out.is_symlink(), "Use a new output directory and at least one report")
    loaded, hashes, labels, identities = [], {}, set(), set()
    for label, folder in results:
        r.require(label.strip() and label not in labels and not out.resolve().is_relative_to(Path(folder).resolve()), "Duplicate label/output inside source")
        labels.add(label)
        value, bound = load_results(label, folder)
        identity = (value["binding"]["execution_sha256"], value["proof"]["review_sha256"])
        r.require(identity not in identities, "Duplicate result snapshot")
        identities.add(identity)
        loaded.append(value)
        hashes.update(bound)
    sources, bound = attach_reviews(loaded, agent_reviews)
    hashes.update(bound)
    phrases, bound = attach_phrase_reviews(loaded, phrase_reviews)
    hashes.update(bound)
    data = {"version": "r5-evaluation-view-v1", "results": loaded, "agent_sources": sources,
            "phrase_review_sources": phrases}
    html = render_html(data)
    unchanged(hashes)
    out.mkdir(mode=0o700)
    partial = out / ".index.html.partial"
    with os.fdopen(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        stream.write(html)
        stream.flush()
        os.fsync(stream.fileno())
    unchanged(hashes)
    partial.rename(out / "index.html")
    return {"path": str(out / "index.html"), "sha256": r.sha(out / "index.html"), "reports": [
        {"label": d["label"], "saved_slots": d["summary"]["finalized_slots"], "planned_slots": d["summary"]["expected_slots"],
         "chat_slots": len(d["rows"]), "agent_reviewed_slots": len(d["agent"]), "started_without_result": d["started_without_result"]} for d in loaded]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--agent-review", action="append", default=[], type=Path)
    parser.add_argument("--phrase-review", action="append", default=[], type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for value in args.results:
        label, separator, folder = value.partition("=")
        r.require(separator and folder, "Use --results LABEL=DIR")
        results.append((label, Path(folder)))
    print(json.dumps(export(results, args.agent_review, args.out, args.phrase_review), indent=2))


if __name__ == "__main__":
    main()
