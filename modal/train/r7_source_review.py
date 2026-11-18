"""Resumable English source triage on Modal; no translation or training admission."""
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import r7_teacher as teacher
app = modal.App("ghana-r7-source-review-private")
image = teacher.image.env({"PYTHONPATH": "/opt"}).add_local_file(
    ROOT / "scripts/r7_english_review_controls.py", "/opt/r7_english_review_controls.py").add_local_file(
    Path(__file__).with_name("r7_teacher.py"), "/opt/r7_teacher.py")
PREFIX = "/understanding-adaptation-r7/source-review/"


def checked_rows(raw, expected_sha, system):
    import hashlib
    import json
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Request inventory changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not 1 <= len(rows) <= 5000 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Invalid explicit source review inventory")
    for offset in range(0, len(rows), 64):
        teacher.validate("r7-source-review-check", "qwen35", rows[offset:offset + 64])
    for row in rows:
        if len(row["messages"]) != 2 or row["messages"][0] != {"role": "system", "content": system}:
            raise ValueError("Changed qualified source-review instructions")
        value = json.loads(row["messages"][-1]["content"])
        if set(value) != {"history", "assistant_target"} or not isinstance(value["assistant_target"], str):
            raise ValueError("Unexpected source-review payload")
    return rows


def terminal_owner(owner):
    def check(graph):
        if not graph:
            raise ValueError("No terminal owner evidence")
        for node in graph:
            if node.status.name not in {"SUCCESS", "FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT"}:
                raise ValueError("Prior owner may still be active")
            if node.children:
                check(node.children)
    check(modal.FunctionCall.from_id(owner).get_call_graph())


@app.function(image=image, cpu=1, memory=2048, timeout=120, retries=0)
def preflight():
    import r7_teacher
    from r7_english_review_controls import controls, VERSION
    return {"teacher_imported": True, "controls": len(controls()),
            "model": r7_teacher.PROFILES["qwen35"], "review_version": VERSION, "gpu_used": False}


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=7200,
              retries=0, max_containers=1, scaledown_window=60,
              volumes={"/cache": teacher.cache.read_only(), "/artifacts": teacher.artifacts})
def review(run_id, requests_sha, qualification_run, previous_owner=None):
    import hashlib
    import json
    import re
    import sys
    import time
    import uuid
    sys.path.insert(0, "/opt")
    from r7_english_review_controls import SYSTEM, VERSION, assess, validate_review
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    claims = modal.Dict.from_name("ghana-r7-source-review-claims", create_if_missing=True)
    if not re.fullmatch(r"r7-source-[a-z0-9-]{3,80}", run_id):
        raise ValueError("Invalid immutable run")
    if not re.fullmatch(r"r7-qwen35-english-review-[a-z0-9-]{3,80}", qualification_run):
        raise ValueError("Invalid qualification identity")
    teacher.artifacts.reload()
    qualification_path = Path("/artifacts/understanding-adaptation-r7/teacher") / qualification_run / "progress.json"
    qualification_raw = qualification_path.read_bytes()
    qualification = json.loads(qualification_raw)
    if not assess(qualification)["qualified_for_english_source_triage_only"]:
        raise ValueError("English source-review qualification did not pass")
    model, revision = teacher.PROFILES["qwen35"]
    if (qualification["model"], qualification["revision"]) != (model, revision):
        raise ValueError("Qualified checkpoint differs")
    folder = Path("/artifacts" + PREFIX) / run_id
    rows = checked_rows((folder / "requests.jsonl").read_bytes(), requests_sha, SYSTEM)
    owner = modal.current_function_call_id()
    binding = {"run_id": run_id, "requests_sha256": requests_sha, "model": model, "revision": revision,
               "review_version": VERSION,
               "qualification_sha256": hashlib.sha256(qualification_raw).hexdigest()}
    claim = {"owner": owner, "binding": binding}
    if not claims.put(run_id, claim, skip_if_exists=True):
        old = claims[run_id]
        if old["binding"] != binding:
            raise ValueError("Run identity changed")
        if old["owner"] != owner:
            if previous_owner != old["owner"]:
                raise ValueError("Explicit prior terminal owner required")
            terminal_owner(old["owner"])
            claims[run_id] = claim
    path = folder / "progress.json"
    report = {**binding, "scope": "english_source_triage_only", "training_eligible": False,
              "twi_validated": False, "clinical_validated": False, "human_reviewed": False,
              "status": "loading", "results": [], "total": len(rows), "attempts": 0,
              "started_at_unix": time.time(), "decoding": {"temperature": 0, "max_tokens": 768, "seed": 42}}
    if path.exists():
        report = json.loads(path.read_text())
        if any(report.get(k) != v for k, v in binding.items()):
            raise ValueError("Saved review binding changed")
        if report["status"] == "complete":
            return report
    if report["attempts"] >= 3:
        raise ValueError("Inspect three attempts before further recovery")
    report["attempts"] += 1
    report["owner"] = owner

    def checkpoint():
        report["elapsed_seconds"] = time.time() - report["started_at_unix"]
        teacher.save(path, report)
        teacher.artifacts.commit()

    checkpoint()
    try:
        source = Path(snapshot_download(model, revision=revision, local_files_only=True,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt"]))
        staged = Path("/tmp") / ("r7-source-review-" + uuid.uuid4().hex)
        report["shards"] = teacher.stage(source, staged)
        checkpoint()
        engine = LLM(model=str(staged), tokenizer=str(staged), language_model_only=True,
            dtype="bfloat16", max_model_len=8192, max_num_seqs=4, gpu_memory_utilization=.90,
            enforce_eager=True, safetensors_load_strategy="lazy", trust_remote_code=False, seed=42)
        tokenizer = engine.get_tokenizer()
        report["status"] = "running"
        done = {r["id"] for r in report["results"]}
        pending = [r for r in rows if r["id"] not in done]
        for offset in range(0, len(pending), 4):
            batch, prompts = [], []
            for row in pending[offset:offset + 4]:
                prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                length = len(tokenizer.encode(prompt, add_special_tokens=False))
                if length + 768 > 8192:
                    report["results"].append({"id": row["id"], "state": "excluded_overlength",
                        "input_tokens": length, "input_sha256": teacher.digest(row), "output": None})
                else:
                    batch.append(row)
                    prompts.append(prompt)
            if batch:
                predictions = engine.generate(prompts, SamplingParams(**report["decoding"]))
                for row, prompt, prediction in zip(batch, prompts, predictions, strict=True):
                    output = prediction.outputs[0]
                    try:
                        review_result = validate_review(output.text, json.loads(row["messages"][-1]["content"]))
                        valid = output.finish_reason == "stop"
                    except (ValueError, TypeError):
                        review_result, valid = None, False
                    report["results"].append({"id": row["id"], "state": "review_returned",
                        "input_sha256": teacher.digest(row), "prompt_sha256": teacher.digest(prompt),
                        "output": output.text, "finish_reason": output.finish_reason,
                        "parsed_review": review_result, "schema_and_source_quotes_valid": valid,
                        "input_tokens": len(prediction.prompt_token_ids), "output_tokens": len(output.token_ids)})
            checkpoint()
            print(json.dumps({"reviewed_or_accounted": len(report["results"]), "total": len(rows)}), flush=True)
        report["status"] = "complete"
        checkpoint()
        return report
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        checkpoint()
        raise
