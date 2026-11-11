"""Read sealed model evidence and source IDs; render a local, dependency-free scorebook."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import html
import json
import re
import unicodedata
import numpy as np

HERE=Path(__file__).resolve().parent
V=("base","r4_response","r5_response")
def read(p): return json.loads(p.read_bytes())
def lines(p): return [json.loads(l) for l in p.open()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s): return " ".join(re.findall(r"[^\W_]+",unicodedata.normalize("NFC",s).casefold()))
def load(name,filename,worker):
    d=HERE/name; p,=d.glob("gpu-*/results.json"); inputs=d/filename; r=read(p)
    assert sha(inputs)==r["cases_sha256"] and sha(HERE/worker)==r["code_sha256"]
    assert set(r["results"])==set(V) and all(r["results"][v]["identity"]["variant"]==v for v in V)
    return lines(inputs),r,{"inputs":sha(inputs),"results":sha(p),"bundle":r["bundle_sha256"],"worker":r["code_sha256"]}

def reference():
    pairs,r,hashes=load("reference-choice-v1","pairs.jsonl","reference_choice_worker.py")
    data={}; stats={}; decisions={}
    for v in V:
        raw=r["results"][v]; z={(p,i,k):(a-b)*(1 if i==k else -1) for p,i,k,a,b in raw["scores"]}
        assert len(z)==len(raw["scores"])==len(pairs)*4 and raw["case_sha256"]==hashes["inputs"]
        decisions[v]=np.array([[min(z[p,i,0],z[p,i,1])>0 for i in (0,1)] for p in range(len(pairs))])
        for p,pair in enumerate(pairs):
            for i,item in enumerate(pair["items"]):
                e=data.setdefault(item["source_id"],{"id":item["source_id"],"domain":pair["domain"],"twi":item["twi"],
                    "reference":item["english"],"foil":pair["items"][1-i]["english"],"margin":{}})
                e["margin"][v]=[z[p,i,0],z[p,i,1]]
        stats[v]={"sentences_both_orders":int(decisions[v].sum()),"pairs_both_sentences_and_orders":int(decisions[v].all(axis=1).sum()),
                  "sentences":len(data),"pairs":len(pairs)}
    differences={}
    for v in V[:-1]:
        a,b=decisions[v],decisions[V[-1]]
        delta=(b.astype(float)-a).mean(axis=1)
        rng=np.random.default_rng(20261111)
        boots=[delta[rng.integers(0,len(delta),size=(500,len(delta)))].mean(axis=1) for _ in range(20)]
        ci=(100*np.quantile(np.concatenate(boots),[.025,.975])).round(2).tolist()
        differences["R5_vs_"+v]={"gains":int((b & ~a).sum()),"losses":int((~b & a).sum()),
            "delta_points":round(100*delta.mean(),2),"paired_95pct_interval_points":ci}
    return list(data.values()),{"results":stats,"differences":differences,"hashes":hashes}

def dialogues():
    cases,r,hashes=load("context-rollout-v1","cases.jsonl","context_rollout_worker.py")
    vocab={norm(c[k]) for c in cases for k in ("changed","unchanged","obsolete")}
    data=[]; stats={}
    for c in cases:
        e={**c,"models":{}}
        for v in V:
            trace=r["results"][v]["trace"][c["id"]]
            assert len(trace)<=4 and all(t["user"]==c["user_turns"][i] for i,t in enumerate(trace))
            checks=[]
            for t,k in [(2,"changed"),(3,"unchanged")]:
                reply=norm(trace[t]["assistant"]) if t<len(trace) else ""
                target=norm(c[k])
                checks.append({"strict":reply==target,"slot":(set(reply.split()) & vocab)=={target},
                    "ended":t<len(trace) and trace[t]["eos"]})
            e["models"][v]={"checks":checks,"trace":trace}
            sub=stats.setdefault(c["language"],{}).setdefault(v,Counter())
            sub.update({"dialogues":1,"both_strict":int(all(j["strict"] for j in checks)),
                "both_slot":int(all(j["slot"] for j in checks)),
                "changed_slot":int(checks[0]["slot"]),"unchanged_slot":int(checks[1]["slot"]),
                "missing_turns":4-len(trace),"turns_hit_token_cap":sum(not t["eos"] for t in trace)})
        data.append(e)
    return data,{"results":stats,"hashes":hashes,"interpretation":"Four project-authored templates, six assignments each in each language. Strict requires the requested bare value, normalized; slot allows prose with the expected value and no other controlled value from any scenario. Neither certifies free prose quality. Incomplete histories remain failures in the denominator. Independent model histories; greedy, 48 new tokens per reply."}

PAGE="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Twi understanding | Checkpoint evidence</title><style>
*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;color:#1c282b;background:#fbfcfa;max-width:1060px;margin:auto;padding:32px 20px}h1{font-size:32px}h2{font-size:22px;margin-top:2em}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px;text-align:left;border-bottom:1px solid #dce2df}thead{color:#355d58}select,input{font:inherit;max-width:100%;padding:5px}summary{cursor:pointer}
article{border-top:1px solid #c9d3d0;padding:12px 0}article p{margin:5px 0}small{color:#475652}label{display:block;margin:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#214d85}section{margin-bottom:30px}</style>
<h1>Twi understanding: checkpoint evidence</h1><p>11 November 2026 · AfriqueQwen 9B, base / R4 / R5 · Private research</p>
<section><h2>Recognising a published meaning</h2><p>1,092 Twi sentences, 546 disjoint same-domain pairs. Each English option appears once in each position. Both orders must be correct. Published references are not gold labels.</p>__TABLE__
<p>R5 minus R4: __DIFF__. Paired bootstrap resamples the 546 pairs, not individual orderings. Only applies to this task and source.</p>
<label>Find a sentence <input id="query" type="search" placeholder="Twi, English or source ID"></label><label>Comparison <select id="filter"><option>All</option><option>R5 passes, R4 misses</option><option>R4 passes, R5 misses</option><option>R5 misses</option></select></label>
<select aria-label="Sentence" id="refs"></select><article id="ref"></article></section>
<section><h2>Following a correction through a conversation</h2><p>24 paired scenarios in Twi and English. Each checkpoint generates all its own replies. Inspect what it heard and what it answered, including cutoffs. The score asks for both the updated and unchanged value, not a rating of prose quality.</p>__DIALOGUE__
<label>Conversation <select id="case"></select></label><article id="trace"></article></section>
<section><h2>What went into training-data repair</h2><p>218 public training-side pairs were reviewed with a qualified API model: 130 judged faithful, 84 marked different, four unclear. This targeted sample is not a corpus-wide error estimate. The teacher did not produce the checkpoint answers above.</p>
<label>Training example <select id="rep"></select></label><article id="repair"></article>
<small>Source: Zakaria et al., 2026, <a href="https://data.mendeley.com/datasets/x3f8w84s7h/1">A Bilingual Corpus for Twi-English Translation</a> (CC BY 4.0). Research annotations are identified separately from published text.</small></section>
<script type="application/json" id="payload">__JSON__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),v=['base','r4_response','r5_response'],names=['Base','R4','R5'],el=id=>document.getElementById(id);
function option(s,id,t){s.add(new Option(t,id))}function line(p,k,t){let a=document.createElement('p'),b=document.createElement('b');b.textContent=k+' ';a.append(b,document.createTextNode(t));p.append(a)}
function refs(){const s=el('refs'),q=el('query').value.trim().toLowerCase(),f=el('filter').selectedIndex;s.replaceChildren();for(const [i,r] of D.refs.entries()){let pass=v.map(k=>Math.min(...r.margin[k])>0);if(f===1&&!(pass[2]&&!pass[1])||f===2&&!(pass[1]&&!pass[2])||f===3&&pass[2])continue;if(![r.id,r.twi,r.reference].join(' ').toLowerCase().includes(q))continue;option(s,i,r.id.split(':').pop()+' · '+r.twi)}ref()}
function ref(){let p=el('ref'),s=el('refs');p.replaceChildren();if(!s.value)return;let r=D.refs[+s.value];line(p,'Twi:',r.twi);line(p,'Published reference:',r.reference);line(p,'Other meaning offered:',r.foil);line(p,'Source:',r.id+' · '+r.domain);v.forEach((k,i)=>line(p,names[i]+':',Math.min(...r.margin[k])>0?'Correct in both orders':'Not correct in both orders'))}
function trace(){const p=el('trace'),c=D.cases[+el('case').value];p.replaceChildren();line(p,'Expected answers:',c.changed+' / '+c.unchanged);c.user_turns.forEach((t,n)=>{line(p,'User '+(n+1)+':',t);v.forEach((k,i)=>{const h=c.models[k].trace[n];line(p,names[i]+':',h?(h.assistant+(h.eos?'':' [output limit]')):'[no valid continuation]')})})}
function rep(){const p=el('repair'),r=D.repair[+el('rep').value];p.replaceChildren();line(p,'Source ID:',r.source.source_id);line(p,'Twi:',r.source.twi);line(p,'Published English:',r.source.english);line(p,'Machine verdict:',r.annotation.verdict);line(p,'Machine rendering:',r.annotation.meaning);if(r.annotation.issue)line(p,'Finding:',r.annotation.issue)}
D.cases.forEach((c,i)=>option(el('case'),i,c.id));D.repair.forEach((r,i)=>option(el('rep'),i,r.source.source_id.split(':').pop()+' · '+r.annotation.verdict));el('refs').onchange=ref;el('query').oninput=refs;el('filter').onchange=refs;el('case').onchange=trace;el('rep').onchange=rep;refs();trace();rep();
</script></html>"""

def table(head,rows):
    return "<table><thead><tr>"+"".join("<th>"+html.escape(x)+"</th>" for x in head)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+html.escape(str(c))+"</td>" for c in r)+"</tr>" for r in rows)+"</tbody></table>"

def main():
    ref,a=reference(); cases,b=dialogues(); assert a["hashes"]["bundle"]==b["hashes"]["bundle"]
    repair=lines(HERE/"public-meaning-repair-v1/reviews.jsonl")
    assert len(ref)==1092 and len(cases)==48 and len(repair)==218
    out=HERE/"results-v1";out.mkdir(mode=0o700,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps({"recognition":a,"conversation":b},indent=2,sort_keys=True)+"\n")
    delta=a["differences"]["R5_vs_r4_response"]
    doc=PAGE.replace("__TABLE__",table(["Checkpoint","Correct in both orders","Both sentences, both orders"],
        [(n,f"{a['results'][v]['sentences_both_orders']} / 1,092",f"{a['results'][v]['pairs_both_sentences_and_orders']} / 546") for n,v in zip(("Base","R4","R5"),V)]))
    doc=doc.replace("__DIFF__",f"{delta['delta_points']:+.2f} percentage points; 95% interval {delta['paired_95pct_interval_points']} points")
    doc=doc.replace("__DIALOGUE__",table(["Both values recalled","Base","R4","R5"],[(lang,*[str(b["results"][k][v]["both_slot"])+" / 24" for v in V]) for lang,k in [("Twi","tw"),("English","en")]]))
    payload=json.dumps({"refs":ref,"cases":cases,"repair":repair},ensure_ascii=False,separators=(",",":")).replace("<","\\u003c").replace(">","\\u003e")
    (out/"index.html").write_text(doc.replace("__JSON__",payload))
    print(json.dumps({"recognition":a["differences"],"rollout":b["results"]},indent=2))

if __name__=="__main__": main()
