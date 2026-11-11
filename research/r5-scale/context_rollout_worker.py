"""Open generation for paired Twi/English dialogue-state probes, each model's own history."""
import hashlib
import json
import os
from pathlib import Path
import sys
import modal

LOCAL=modal.is_local()
HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/"understanding-r5-20261109/inferencebundle-v1" if LOCAL else Path("/bundle")
CASES=HERE/"context-rollout-v1/cases.jsonl" if LOCAL else Path("/test/cases.jsonl")
def sha(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
CH="749e3c388a382700c91e5e736590f67ae5e0be7eefb9569b09ce8f4627a2b5d2"
assert sha(CASES)==CH
RH=sha(Path(__file__))
BH=sha(BUNDLE/"manifest.json")
app=modal.App("ghana-twi-context-rollout-research")
versions=json.loads((BUNDLE/"manifest.json").read_bytes())["runtime"]
image=modal.Image.debian_slim(python_version="3.11").pip_install(*[k+"=="+v for k,v in versions.items()]).env({
    "HF_HOME":"/cache/hf","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTHONPATH":"/worker:/bundle/code",
    "PYTHONDONTWRITEBYTECODE":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8","TOKENIZERS_PARALLELISM":"false","DO_NOT_TRACK":"1"})
if LOCAL:
    image=image.add_local_dir(str(BUNDLE),"/bundle").add_local_file(str(CASES),"/test/cases.jsonl").add_local_file(__file__,"/worker/context_rollout_worker.py")
cache=modal.Volume.from_name("ghana-health-understanding-hf-cache")
artifacts=modal.Volume.from_name("ghana-health-understand-train")
output=modal.Volume.from_name("ghana-health-understanding-measurements")

@app.function(image=image,gpu="H100",memory=49152,cpu=4,timeout=7200,retries=0,volumes={
    "/cache":cache.read_only(),"/artifacts":artifacts.read_only(),"/scores":output},include_source=False)
def measure(bundle_sha:str, code_sha:str):
    import torch
    from transformers import GenerationConfig
    sys.path.insert(0,"/bundle/code")
    import r5_inference
    from afrique_instruction_baseline_core import prompt_ids
    from check_afrique_training_contract import EOS,PAD
    assert (BH,RH)==(bundle_sha,code_sha)
    rt=r5_inference.Runtime(BUNDLE,BH)
    cases=[json.loads(l) for l in CASES.read_text().splitlines()]
    dest=Path("/scores")/CH/BH/RH
    dest.mkdir(parents=True,exist_ok=True)
    conf=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.0,max_new_tokens=48,
                          eos_token_id=EOS,pad_token_id=PAD,bos_token_id=None,use_cache=True)
    parent=dest.parent/"cb64bd0dadd4f3a3d9a4a8367a7ae247c09e878ed09d5b3e75feaf5c1e9121ad"
    previous_hashes={"r4_response-0.json":"21b7dd8df7b42c9110658e58566eda1ef1332ea2e7790939271bd2382b7ada0f",
        "base-0.json":"9d19afdfaa095d43add8f4b8fc081934cde96344b407821abdefa416ab054585",
        "base-1.json":"2971a0773c19a1494d0521240a6b65928b7b351b4458d92c66b21ca7c492b7d8",
        "base-2.json":"2e943bd4f61ee0a21e3989d1dc59a80c71e51d30242a5068102fb37f91f9dd51",
        "base-3.json":"0da95d48f91a668bfb929e715e8670a586c73a45aad921ee9bd434edf68f14a3"}
    results={}
    for variant in ("base","r4_response","r5_response"):
        trace={r["id"]:[] for r in cases}
        with rt.select(variant),torch.inference_mode():
            for turn in range(4):
                path=dest/(variant+f"-{turn}.json")
                if path.name in previous_hashes and not path.exists():
                    original=parent/path.name
                    assert sha(original)==previous_hashes[path.name]
                    path.write_bytes(original.read_bytes());output.commit()
                if path.exists():
                    saved=json.loads(path.read_bytes())
                    assert saved["identity"]==rt.identity(variant)
                    trace=saved["trace"]
                    continue
                pending=[]
                for c in cases:
                    previous=trace[c["id"]]
                    if len(previous)!=turn or any(not x["assistant"].strip() for x in previous): continue
                    history=[]
                    for h in previous:
                        history.extend([{"role":"user","content":h["user"]},{"role":"assistant","content":h["assistant"]}])
                    user=c["user_turns"][turn]
                    history.append({"role":"user","content":user})
                    try:
                        ids=prompt_ids(rt.tokenizer,history,{"context_tokens":4096,"max_new_tokens":48})
                    except ValueError as exc:
                        if not previous or str(exc) not in ("Invalid Unicode in message","Reserved source token"): raise
                        previous[-1]["cannot_continue"]=str(exc)
                        continue
                    pending.append((len(ids),c["id"],user,ids))
                pending.sort()
                for start in range(0,len(pending),8):
                    group=pending[start:start+8]
                    width=max(z[0] for z in group)
                    x=torch.tensor([[PAD]*(width-z[0])+z[3] for z in group],device="cuda")
                    mask=torch.tensor([[0]*(width-z[0])+[1]*z[0] for z in group],device="cuda")
                    outs=rt.model.generate(input_ids=x,attention_mask=mask,generation_config=conf)
                    for (_,cid,user,ids),tokens in zip(group,outs[:,width:].tolist(),strict=True):
                        eos=EOS in tokens
                        if eos: tokens=tokens[:tokens.index(EOS)]
                        text=rt.tokenizer.decode(tokens,skip_special_tokens=True)
                        trace[cid].append({"user":user,"assistant":text,"output_ids":tokens,"eos":eos,
                            "prompt_ids_sha256":r5_inference.digest(ids)})
                saved={"identity":rt.identity(variant),"trace":trace}
                tmp=path.with_suffix(".pending")
                tmp.write_text(json.dumps(saved,sort_keys=True));os.replace(tmp,path);output.commit()
                print("Committed",variant,turn+1,flush=True)
        results[variant]={"identity":rt.identity(variant),"trace":trace}
    return {"cases_sha256":CH,"code_sha256":RH,"bundle_sha256":BH,"results":results,
            "max_new_tokens":48,"raw":True,"batch_size":8,"volume":"ghana-health-understanding-measurements",
            "directory":str(dest.relative_to("/scores"))}

@app.local_entrypoint()
def launch():
    p=HERE/"context-rollout-v1"/("gpu-"+RH[:12])
    p.mkdir(exist_ok=True,mode=0o700)
    receipt=p/"call.json"
    if receipt.exists():
        call=modal.FunctionCall.from_id(json.loads(receipt.read_bytes())["call_id"])
    else:
        call=measure.spawn(BH,RH)
        receipt.write_text(json.dumps({"call_id":call.object_id,"code_sha256":RH,"bundle_sha256":BH,"cases_sha256":CH}))
    print("CALL",call.object_id,flush=True)
    result=call.get()
    assert (result["cases_sha256"],result["code_sha256"],result["bundle_sha256"])==(CH,RH,BH)
    (p/"results.json").write_text(json.dumps(result,sort_keys=True))
    print("SAVED",p/"results.json",flush=True)
