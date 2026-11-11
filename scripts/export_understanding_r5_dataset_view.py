#!/usr/bin/env python3
"""Export a private, offline view of the sealed R5 pack and completed consumption.

Usage: python3 scripts/export_understanding_r5_dataset_view.py --pack P/pack-v1 \
    --consumption P/train-v1/completed/consumption.json --out P/dataset-view-v1
Requires the receipt's sibling status.json and execution.json. No model, network,
tokenizer execution, registry writes, or dereferencing of original source paths.
"""
import argparse
import base64
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

import run_understanding_r5 as r5

CLASSES = {"responses": "Responses", "meaning": "Meaning", "language": "Language",
           "english_retention": "English retention"}
AMOUNTS = ("unique_rows", "row_exposures", "input_tokens", "effective_target_tokens")


def record_key(ref):
    return tuple(ref[k] for k in ("dataset_id", "dataset_version", "record_id"))


def training_hashes(rows, schedule):
    # This helper validates/hashes integer arrays only; importing it does not load Torch.
    path = r5.ROOT / "modal/train/understanding_training_core.py"
    spec = importlib.util.spec_from_file_location("r5_dataset_view_core", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core._inputs(rows, schedule)[3:]


def load_snapshot(pack, consumption):
    pack, consumption = Path(pack).resolve(strict=True), Path(consumption).absolute()
    completed = consumption.parent.resolve(strict=True)
    receipt_path = r5.safe_file(completed, consumption.name)
    proof_paths = [r5.safe_file(pack, n) for n in ("manifest.json", "PREPARED.json")]
    proof_paths += [receipt_path] + [r5.safe_file(completed, n) for n in ("status.json", "execution.json")]
    hashes = {p: r5.sha(p) for p in proof_paths}
    manifest, rows, stages = r5.verify_pack(pack)
    hashes.update({r5.safe_file(pack, name): checksum for name, checksum in manifest["artifacts"].items()})
    receipt, status, execution = (r5.read(p) for p in proof_paths[2:])
    data_sha, execution_sha = hashes[pack / "manifest.json"], hashes[completed / "execution.json"]
    r5.require(status["state"] == "completed" and status["training_completed"] is True
               and status["mode"] == execution["mode"] == receipt["mode"] == "train",
               "Completed training evidence required")
    r5.require(status["consumption_sha256"] == hashes[receipt_path]
               and status["data_sha256"] == receipt["data_sha256"] == execution["data_sha256"] == data_sha
               and status["execution_sha256"] == receipt["execution_sha256"] == execution_sha
               and status["run_id"] == r5.run_id(receipt["run_id"]), "Completion identity differs")
    r5.require(len(status["stages"]) == 1, "Expected one completed stage")
    terminal, stage, checkpoint = status["stages"][0], stages[0], receipt["checkpoint"]
    encoded_sha, schedule_sha = training_hashes(rows, stage["schedule"])
    r5.require(terminal["completion_verified"] is True and terminal["stage"] == stage["name"]
               and terminal["run_sha256"] == execution_sha and terminal["data_sha256"] == data_sha
               and terminal["encoded_rows_sha256"] == encoded_sha
               and terminal["schedule_sha256"] == schedule_sha, "Completed stage differs")
    r5.require(set(checkpoint) == {"directory", "contract_sha256", "manifest_sha256", "state_sha256"}
               and all(isinstance(checkpoint[k], str) and re.fullmatch(r"[a-f0-9]{64}", checkpoint[k])
                       for k in ("contract_sha256", "manifest_sha256", "state_sha256"))
               and checkpoint["directory"] == terminal["checkpoint"]
               and checkpoint["contract_sha256"] == terminal["checkpoint_run_sha256"]
               and receipt["completed_updates"] == terminal["checkpoint_completed_updates"],
               "Consumption checkpoint differs")
    expected = {"run_id": status["run_id"], **r5.consumption_receipt(
        rows, stage, manifest["optimization"]["grad_accum"], terminal["position"],
        snapshots=stage["admitted_dataset_snapshots"], mode="train", execution_sha=execution_sha,
        data_sha=data_sha, checkpoint=checkpoint)}
    r5.require(expected["training_complete"] and r5.digest(expected) == r5.digest(receipt),
               "Receipt differs from exact committed schedule accounting")
    witnesses = r5.json_rows(pack / "source-witnesses.jsonl")
    sources, datasets = {}, []
    # verify_pack already checks admitted_sources: exact hashes, source IDs and prefixes.
    for ref in stage["admitted_dataset_snapshots"]:
        source = r5.read(r5.safe_file(pack, ref["manifest_path"]))
        datasets.append({**ref, **source})
        for record in r5.json_rows(r5.safe_file(pack, source["records_path"])):
            sources[(source["dataset_id"], source["dataset_version"], record["id"])] = record
    selected = {}
    for consumed in receipt["rows"]:
        i, ref = consumed["row_index"], consumed["dataset_ref"]
        key, row, witness = record_key(ref), rows[i], witnesses[i]
        r5.require(row["collection"] in CLASSES, "Unknown dataset class")
        item = selected.setdefault(key, {"class": row["collection"], "ref": ref,
                                        "record": sources[key], "counts": Counter(), "views": []})
        r5.require(item["class"] == row["collection"], "Record spans dataset classes")
        item["counts"].update({k: consumed[k] for k in AMOUNTS})
        item["views"].append({"consumption": consumed, "objective": row["objective"],
                              "message_count": len(witness.get("messages", [])),
                              "metadata": witness["metadata"], "original_ref": witness["original_ref"]})
    r5.require(len(selected) == receipt["consumed"]["unique_dataset_records"], "Selected record count differs")
    result = {"version": "r5-dataset-view-v1", "classes": CLASSES, "run_id": receipt["run_id"],
              "totals": receipt["consumed"], "datasets": datasets,
              "proof": {"pack_manifest_sha256": data_sha, "consumption_sha256": hashes[receipt_path],
                        "status_sha256": hashes[completed / "status.json"], "execution_sha256": execution_sha,
                        "checkpoint": checkpoint, "completed_updates": receipt["completed_updates"],
                        "checkpoint_proof_scope": "Local completion receipt; remote checkpoint bytes not re-read",
                        "registry_mutated": False, "new_certification": False},
              "records": [selected[k] for k in sorted(selected)]}
    check_unchanged(hashes)
    return result, hashes


def check_unchanged(hashes):
    r5.require(all(not p.is_symlink() and r5.sha(p) == value for p, value in hashes.items()),
               "Input changed during export")


def escaped_json(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character in "<>&\u2028\u2029":
        text = text.replace(character, "\\u%04x" % ord(character))
    return text


STYLE = """
*{box-sizing:border-box;letter-spacing:0}body{margin:0;color:#202926;background:#fff;font:14px/1.5 system-ui,sans-serif}
header{padding:20px 28px 0;border-top:5px solid #24715d}h1{font-size:25px;margin:0}h2{font-size:18px;margin:0 0 12px}h3{font-size:14px;margin:20px 0 8px}
p{margin:6px 0 12px}.muted,dt{color:#586560}small{font-size:12px}button,input,select{font:inherit;color:inherit;border:1px solid #b9c5bf;border-radius:4px;background:#fff;padding:8px}
button{cursor:pointer}button:hover{background:#eaf2ee}button:disabled{opacity:.4;cursor:default}:focus-visible{outline:3px solid #388870;outline-offset:2px}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;padding:14px 0}.metric{border-left:3px solid #d5e4dc;padding-left:12px}.metric strong{display:block;font-size:23px;font-weight:600}
.metric small{display:block}.metric:nth-child(3){border-color:#c99c44}.metric:nth-child(4){border-color:#7d97b8}
.toolbar{display:flex;gap:16px;padding:16px 28px;border-block:1px solid #dde4df;background:#f7f9f7}.toolbar label{display:grid;gap:4px}.search{flex:1}.toolbar input{width:100%}
main{display:grid;grid-template-columns:minmax(320px,44%) minmax(0,1fr);min-height:65vh}.master{padding:16px 20px 20px 28px;border-right:1px solid #dde4df;min-width:0}
.table-scroll{overflow:auto;max-height:65vh;border-block:1px solid #dce3df}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:9px 8px;text-align:left;vertical-align:top;border-bottom:1px solid #e3e8e5;overflow-wrap:anywhere}
th{font-size:12px;color:#586560;background:#f7f9f7;position:sticky;top:0}th:first-child{width:56%}th:last-child{width:16%}.row-button{border:0;padding:0;text-align:left;background:none;width:100%;color:#185b49;font-weight:600}
tr.selected{background:#eaf3ee}td small{display:block;max-height:3em;overflow:hidden}td .record-id{white-space:nowrap;text-overflow:ellipsis}td:last-child{font-variant-numeric:tabular-nums}.pager{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:12px 0}.pager button{width:38px;height:36px;padding:0;font-size:20px}
#detail{padding:20px 28px;min-width:0;max-height:83vh;overflow:auto}#detail h2{overflow-wrap:anywhere}dl{display:grid;grid-template-columns:140px minmax(0,1fr);gap:5px 12px;margin:10px 0}dd{margin:0;overflow-wrap:anywhere}
.message{border-left:3px solid #dce4df;padding:6px 12px;margin:8px 0;background:#fafbf9}.message.assistant{border-color:#7f9f92}.message b{font-size:12px;color:#52635b}
.text,pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:normal;margin:4px 0}pre{font-size:12px}.fields{border-bottom:1px solid #e3e8e5;padding-bottom:12px}summary{cursor:pointer;color:#315d4f;padding:10px 0}details{border-top:1px solid #e3e8e5}footer{padding:16px 28px;background:#f7f9f7;border-top:1px solid #dce4df;overflow-wrap:anywhere}
@media(max-width:760px){header{padding:16px 16px 0}.toolbar{padding:12px 16px;flex-wrap:wrap}.toolbar label{width:100%}.metrics{gap:10px}.metric strong{font-size:19px}.metric{padding-left:7px}.metric small{font-size:11px}main{grid-template-columns:minmax(0,1fr)}.master{padding:16px;border-right:0}.table-scroll{max-height:350px}th:first-child{width:50%}th:last-child{width:22%}#detail{padding:16px;max-height:none;border-top:1px solid #dde4df}dl{grid-template-columns:110px minmax(0,1fr)}footer{padding:16px}}
@media(max-width:430px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(min-width:761px) and (max-width:1100px){#detail .metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

SCRIPT = r"""
'use strict';
const data=JSON.parse(document.getElementById('snapshot').textContent), $=id=>document.getElementById(id);
const fmt=n=>n.toLocaleString('en-US'), human=s=>String(s).replaceAll('_',' ');
const node=(tag,text,parent)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(parent)parent.append(e);return e;};
const pretty=v=>typeof v==='string'?v:JSON.stringify(v,null,2);
const metadata=r=>r.metadata||{}, review=r=>r.review||{};
const origin=r=>r.origin||metadata(r).origin||'Not recorded';
const origins={'source_grounded_ai_translated_and_agent_corrected':'Source translation + agent correction','source_grounded_agent_translated_and_separately_reviewed':'Source translation + separate agent review',
 'agent_revised_or_retained_source':'Agent-revised / retained source','agent_translated_source_with_synthetic_user_spelling':'Translation + synthetic user spelling',
 'project_authored_agent_translated_and_agent_corrected':'Project-authored + agent correction','project_authored_agent_translated_and_separately_reviewed':'Project-authored + separate agent review',
 'project_authored_ai_translated':'Project-authored AI translation','source_english_separately_reviewed_for_r5':'English source + separate agent review',
 'compiled_parallel_text':'Parallel text','speech_transcript':'Speech transcript','source_preserved':'Source-preserved'};
const sourceLabel=r=>origins[origin(r)]||human(origin(r));
const status=r=>review(r).status||r.quality_status||metadata(r).screening||'Not recorded';
const summary=r=>r.text||(r.messages||[]).find(m=>m.role==='user')?.content||r.id;
const amounts=[['unique_dataset_records','Registered records'],['unique_rows','Training views'],['row_exposures','Actual exposures'],['effective_target_tokens','Supervised tokens']];
function metrics(parent,counts){parent.replaceChildren();for(const [k,label] of amounts){const e=node('div',undefined,parent);e.className='metric';node('strong',fmt(counts[k]||0),e);node('small',label,e);}}
function fields(parent,values){const dl=node('dl',undefined,parent);for(const [k,v] of Object.entries(values)){node('dt',human(k),dl);const text=Array.isArray(v)&&v.every(x=>typeof x==='string')?v.join('\n'):pretty(v===undefined?'Not recorded':v);node('dd',text,dl).className='text';}}
function conversation(parent,title,messages){if(!Array.isArray(messages)||!messages.length)return;node('h3',title,parent);for(const m of messages){const e=node('div',undefined,parent);e.className='message '+(m.role==='assistant'?'assistant':'');node('b',human(m.role),e);node('div',m.content,e).className='text';}}
function collapsed(parent,title,value){const d=node('details',undefined,parent);node('summary',title,d);node('pre',pretty(value),d);}
let current='responses', page=0, selected=null, filtered=[], size=25;
const searchable=new Map(data.records.map(r=>[r,JSON.stringify(r).toLocaleLowerCase()]));
function detail(item){const pane=$('detail');pane.replaceChildren();if(!item){node('p','No matching records.',pane);return;}const r=item.record,m=metadata(r),v=review(r);
 node('h2',data.classes[item.class]+' record',pane);node('p',sourceLabel(r),pane).className='muted';
 fields(pane,{'Review status':human(status(r)),'Human reviewed':r.human_reviewed??v.human_reviewed??m.human_reviewed??'Not recorded'});
 const counts=node('div',undefined,pane);counts.className='metrics';metrics(counts,{...item.counts,unique_dataset_records:1});
 conversation(pane,'Registered messages',r.messages);if(r.text){node('h3','Registered text',pane);node('div',r.text,pane).className='text';}
 conversation(pane,'English source / reference',r.english_messages);conversation(pane,'Original English (before recorded amendments)',r.original_english_messages);
 if(m.meaning_english){node('h3','English meaning reference',pane);node('div',m.meaning_english,pane).className='text';}
 if(!r.english_messages?.length){node('h3','English source / reference',pane);node('p',item.class==='english_retention'?'English source is shown in registered messages.':'English text not present as a separate field in the sealed record.',pane);}
 if(r.english_provenance){fields(pane,Object.fromEntries(Object.entries(r.english_provenance).filter(([k])=>!k.includes('sha256'))));}
 if(r.amendment_reason){fields(pane,{'Amendment reason':r.amendment_reason});}if(r.english_amendments?.length)collapsed(pane,'Recorded English amendments',r.english_amendments);
 node('h3','Source and review provenance',pane);
 fields(pane,{'Record ID':r.id,'Source ID':r.source_id,'Origin source IDs':r.source_ids||r.canonical_source_id||m.source_id||r.source_id,
  'Origin':origin(r),'Source type':r.record_type||m.record_type||(r.messages?'Conversation':'Text'),
  'Source family':r.source_family||m.source_family||r.english_provenance?.repo||'Not recorded',
  'Language':r.language||m.language||[m.source_language,m.target_language].filter(Boolean).join(' / ')||'Not recorded',
  'Review kind':v.critic_kind||r.quality_status||m.screening||v.status||'Not recorded','Reviewer':v.reviewer||'Not recorded',
  'Semantic certification':r.semantic_certification??m.semantic_certification??'Not recorded','Clinical certification':r.clinical_certification??'Not recorded',
  'Input tokens (exposures)':item.counts.input_tokens,'Dataset':item.ref.dataset_id,'Dataset version':item.ref.dataset_version});
 node('h3','Original reference',pane);fields(pane,Object.fromEntries(Object.entries(r.original_ref||{}).filter(([k,v])=>!k.includes('sha256')&&typeof v!=='object')));
 if(!r.original_ref)node('p','Not present in registered record; training-view references are retained below.',pane);
 if(v.source_review||r.source_quality_review)fields(pane,{'Source review':v.source_review||r.source_quality_review});
 if(Object.keys(v).length)collapsed(pane,'Recorded review details',v);
 node('h3','Consumed training views',pane);for(const view of item.views){const c=view.consumption;fields(pane,{'View':c.id,'Objective':human(view.objective),
  'Messages in prefix':view.message_count||'Not applicable','Actual exposures':c.row_exposures,'Supervised tokens (exposures)':c.effective_target_tokens});}
 collapsed(pane,'Technical provenance / complete registered record',{dataset_ref:item.ref,registered_record:r,training_views:item.views});pane.scrollTop=0;
}
function render(){const q=$('search').value.toLocaleLowerCase().trim();filtered=data.records.filter(r=>r.class===current&&searchable.get(r).includes(q));
 page=Math.min(page,Math.max(0,Math.ceil(filtered.length/size)-1));const visible=filtered.slice(page*size,(page+1)*size);
 if(!visible.includes(selected))selected=visible[0]||null;
 const body=$('rows');body.replaceChildren();for(const item of visible){const r=item.record,tr=node('tr',undefined,body);if(item===selected)tr.className='selected';
  const td=node('td',undefined,tr),text=summary(r),b=node('button',text.length>130?text.slice(0,130)+'...':text,td);b.className='row-button';b.setAttribute('aria-pressed',String(item===selected));b.onclick=()=>{selected=item;render();if(matchMedia('(max-width:760px)').matches)$('detail').scrollIntoView();};
  node('small',r.id,td).className='record-id';const source=node('td',sourceLabel(r),tr);node('small',human(status(r)),source);node('td',fmt(item.counts.row_exposures),tr);}
 if(!visible.length){const td=node('td','No matching records.',node('tr',undefined,body));td.colSpan=3;}
 $('page').textContent=filtered.length?`${fmt(page*size+1)}-${fmt(Math.min((page+1)*size,filtered.length))} of ${fmt(filtered.length)} records`:'0 records';
 $('previous').disabled=page===0;$('next').disabled=(page+1)*size>=filtered.length;
 const c=data.totals.by_class[current]||{};$('class-counts').textContent=`${fmt(c.unique_dataset_records||0)} records / ${fmt(c.unique_rows||0)} views / ${fmt(c.row_exposures||0)} exposures / ${fmt(c.effective_target_tokens||0)} supervised tokens`;
 detail(selected);
}
metrics($('totals'),data.totals);$('run').textContent=data.run_id;
for(const [key,label] of Object.entries(data.classes)){const o=node('option',label,$('class'));o.value=key;}
$('class').value=current;$('class').onchange=()=>{current=$('class').value;page=0;selected=null;render();};
$('search').oninput=()=>{page=0;render();};$('previous').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};
fields($('proof'),data.proof);collapsed($('proof'),'Registered dataset manifests',data.datasets);render();
"""


def render_html(snapshot):
    script_hash = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'sha256-{script_hash}'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; font-src 'none'; form-action 'none'; base-uri 'none'; object-src 'none'">
<title>R5 training dataset</title><style>{STYLE}</style></head><body>
<header><h1>R5 training dataset</h1><p class="muted">Private research / completed training / <span id="run"></span></p><div id="totals" class="metrics" aria-label="Completed run totals"></div></header>
<div class="toolbar"><label>Dataset class<select id="class"></select></label><label class="search">Search records<input id="search" type="search" autocomplete="off" spellcheck="false"></label></div>
<main><section class="master" aria-label="Records"><p id="class-counts" class="muted"></p><div class="table-scroll"><table><thead><tr><th>Record</th><th>Origin / review</th><th>Exposures</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="pager"><button id="previous" title="Previous page" aria-label="Previous page">&#8592;</button><span id="page" role="status"></span><button id="next" title="Next page" aria-label="Next page">&#8594;</button></div></section>
<section id="detail" aria-label="Selected record"></section></main>
<footer>Selected registered records only. Views and repeated exposures are not additional records. Supervised tokens include repeated exposures and exclude masked context.<br>
Recorded review metadata; no new human, native-language, or clinical certification. Private snapshot view, not a new dataset.<details><summary>Snapshot verification</summary><div id="proof"></div></details></footer>
<noscript>JavaScript is required to display the embedded records.</noscript>
<script id="snapshot" type="application/json">{escaped_json(snapshot)}</script><script>{SCRIPT}</script></body></html>"""


def export(pack, consumption, out):
    out = Path(out).absolute()
    r5.require(not out.exists() and not out.is_symlink(), "Output directory already exists")
    resolved = out.resolve()
    r5.require(not any(resolved.is_relative_to(Path(p).resolve()) for p in (pack, Path(consumption).parent)),
               "Output must be outside sealed inputs")
    snapshot, hashes = load_snapshot(pack, consumption)
    html = render_html(snapshot)
    check_unchanged(hashes)
    out.mkdir(mode=0o700)  # A new private directory only; never overwrite another export.
    partial = out / ".index.html.partial"
    with os.fdopen(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        stream.write(html)
        stream.flush()
        os.fsync(stream.fileno())
    check_unchanged(hashes)
    partial.rename(out / "index.html")
    return {"path": str(out / "index.html"), "sha256": r5.sha(out / "index.html"),
            "counts": {k: v for k, v in snapshot["totals"].items() if not k.startswith("by_")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--consumption", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New private directory; never overwritten")
    args = parser.parse_args()
    print(json.dumps(export(args.pack, args.consumption, args.out), indent=2))


if __name__ == "__main__":
    main()
