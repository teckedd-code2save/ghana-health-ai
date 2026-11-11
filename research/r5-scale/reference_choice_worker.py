"""Private Modal benchmark. No generation, upstream API calls, training or deployment."""
import hashlib
import json
import os
from pathlib import Path
import sys

import modal

LOCAL=modal.is_local()
HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/"understanding-r5-20261109/inferencebundle-v1" if LOCAL else Path("/bundle")
CASES=HERE/"reference-choice-v1/pairs.jsonl" if LOCAL else Path("/test/pairs.jsonl")
def digest(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

BH=digest(BUNDLE/"manifest.json")
CH="9d15b0686f44686cb6806f7e56a879e660c33270f93230fe432920f2ff687a4d"
assert digest(CASES)==CH
RH=digest(Path(__file__))
app=modal.App("ghana-twi-reference-choice-research")
versions=json.loads((BUNDLE/"manifest.json").read_bytes())["runtime"]
image=modal.Image.debian_slim(python_version="3.11").pip_install(*[k+"=="+v for k,v in versions.items()]).env({
    "HF_HOME":"/cache/hf","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
    "PYTHONPATH":"/worker:/bundle/code","PYTHONDONTWRITEBYTECODE":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8",
    "TOKENIZERS_PARALLELISM":"false","DO_NOT_TRACK":"1"})
if LOCAL:
    image=image.add_local_dir(str(BUNDLE),"/bundle").add_local_file(str(CASES),"/test/pairs.jsonl").add_local_file(__file__,"/worker/reference_choice_worker.py")
cache=modal.Volume.from_name("ghana-health-understanding-hf-cache")
artifacts=modal.Volume.from_name("ghana-health-understand-train")
output=modal.Volume.from_name("ghana-health-understanding-measurements",create_if_missing=True)
VARIANTS=("base","r4_response","r5_response")

@app.function(image=image,gpu="H100",memory=49152,cpu=4,timeout=7200,retries=0,volumes={
    "/cache":cache.read_only(),"/artifacts":artifacts.read_only(),"/scores":output},include_source=False)
def measure(bundle_sha:str, code_sha:str):
    import collections
    import math
    import torch
    assert bundle_sha==BH and code_sha==RH
    sys.path.insert(0,"/bundle/code")
    import r5_inference
    from afrique_instruction_baseline_core import prompt_ids
    rt=r5_inference.Runtime(BUNDLE,BH)
    labels=[rt.tokenizer.encode(c,add_special_tokens=False) for c in ("A","B")]
    assert all(len(x)==1 for x in labels) and labels[0]!=labels[1]
    ab=[x[0] for x in labels]
    pairs=[json.loads(x) for x in CASES.read_text().splitlines()]
    tasks=[]
    for pairno,pair in enumerate(pairs):
        for side, item in enumerate(pair["items"]):
            for reverse in (0,1):
                options=pair["items"][::(-1 if reverse else 1)]
                question=("Which English sentence better translates the Twi text? Reply with A or B only.\n"
                          "Twi: "+item["twi"].strip()+"\nA: "+options[0]["english"].strip()+"\nB: "+options[1]["english"].strip())
                ids=prompt_ids(rt.tokenizer,[{"role":"user","content":question}],{"context_tokens":4096,"max_new_tokens":1})
                tasks.append((pairno,side,reverse,ids))
    assert len(tasks)==2184
    groups=collections.defaultdict(list)
    for idx,task in enumerate(tasks): groups[len(task[-1])].append(idx)
    dest=Path("/scores")/CH/BH/RH
    dest.mkdir(parents=True,exist_ok=True)
    results={}
    for variant in VARIANTS:
        file=dest/(variant+".json")
        if file.exists():
            old=json.loads(file.read_bytes())
            assert old["identity"]==rt.identity(variant) and len(old["scores"])==len(tasks)
            results[variant]=old
            continue
        scores=[None]*len(tasks)
        with rt.select(variant),torch.inference_mode():
            for size, indexes in sorted(groups.items()):
                for start in range(0,len(indexes),16):
                    ix=indexes[start:start+16]
                    x=torch.tensor([tasks[i][-1] for i in ix],device="cuda")
                    # Equal length, no padding, one next-token position per example.
                    logits=rt.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits
                    assert logits.shape[:2]==(len(ix),1)
                    vals=logits[:,0,ab].float().cpu().tolist()
                    for i,(a,b) in zip(ix,vals,strict=True):
                        assert math.isfinite(a) and math.isfinite(b)
                        scores[i]=[*tasks[i][:3],a,b]
        assert all(x is not None for x in scores)
        record={"identity":rt.identity(variant),"case_sha256":CH,"code_sha256":RH,"choice_token_ids":ab,"scores":scores}
        temp=file.with_suffix(".pending")
        temp.write_text(json.dumps(record,sort_keys=True));os.replace(temp,file);output.commit()
        results[variant]=record
        print("Committed",variant,len(scores),flush=True)
    return {"volume": "ghana-health-understanding-measurements","directory":str(dest.relative_to("/scores")),
            "cases_sha256":CH,"code_sha256":RH,"bundle_sha256":BH,"results":results}

@app.local_entrypoint()
def launch():
    p=HERE/"reference-choice-v1"/("gpu-"+RH[:12])
    p.mkdir(exist_ok=True,mode=0o700)
    receipt=p/"call.json"
    if receipt.exists():
        call=modal.FunctionCall.from_id(json.loads(receipt.read_bytes())["call_id"])
    else:
        call=measure.spawn(BH,RH)
        receipt.write_text(json.dumps({"call_id":call.object_id,"code_sha256":RH,"bundle_sha256":BH,"cases_sha256":CH}))
    print("CALL",call.object_id,flush=True)
    result=call.get()
    assert result["cases_sha256"]==CH and result["code_sha256"]==RH and result["bundle_sha256"]==BH
    (p/"results.json").write_text(json.dumps(result,sort_keys=True))
    print("SAVED",p/"results.json",flush=True)
