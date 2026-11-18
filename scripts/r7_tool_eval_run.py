"""Prepare, launch explicitly, collect and replay matched R6/final-R7 tool runs.

Commands: prepare, stage, launch, collect, score, compare. CPU commands never
contact Modal. Launch is detached, single H100, zero warm containers, no retries.
An existing/uncertain receipt is never automatically resubmitted.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys

import r7_tool_eval as ev

ROOT = Path(__file__).resolve().parents[1]
APP = "ghana-r7-tool-eval-private"
VOLUME = "ghana-health-understand-train"
CACHE = "ghana-health-understanding-hf-cache"
CLAIMS = "ghana-r7-tool-eval-invocations"
PREFIX = "understanding-adaptation-r7/tool-eval"
CODE = ("r7_tool_eval.py", "r7_tool_eval_run.py", "r7_tool_world.py",
        "prepare_r7_tool_candidates.py", "r7_encoding.py")
SOURCE_FILES = ("manifest.json", "PREPARED.json", "candidates/dev.jsonl", "scenarios.jsonl",
                "executions.jsonl", "source-registry.jsonl", "tools.json")


def inventory(folder):
    return {p.relative_to(folder).as_posix(): ev.sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def seal(folder, manifest):
    ev.save(folder / "manifest.json", {**manifest, "files": inventory(folder)})
    checksum = ev.sha(folder / "manifest.json")
    ev.save(folder / "SEALED.json", {"manifest_sha256": checksum})
    return checksum


def derive_cases(source):
    import prepare_r7_tool_candidates as candidates
    rows = ev.read_rows(source / "candidates/dev.jsonl")
    receipts = {r["id"]: r for r in ev.read_rows(source / "executions.jsonl")}
    scenarios = {r["id"]: r for r in ev.read_rows(source / "scenarios.jsonl")}
    registry = {r["id"]: r for r in ev.read_rows(source / "source-registry.jsonl")}
    cases, references = [], []
    for row in rows:
        scenario = scenarios[row["scenario_id"]]
        group = registry[row["source_group_id"]]
        ev.require(row["split"] == scenario["split"] == group["split"] == "dev", "Non-dev source")
        ev.require(row["source_group_id"] == scenario["source_group_id"], "Scenario group differs")
        ev.require(row["tools"] == ev.world.tools(), "Source schemas differ")
        for binding in row["metadata"]["tool_bindings"]:
            receipt = receipts[binding["execution_id"]]
            ev.require(ev.world.sha256_json(receipt) == binding["execution_record_sha256"], "Source execution pin differs")
            actual = ev.world.execute(receipt["execution"]["call"]["name"], receipt["arguments"], receipt["context"])
            ev.require(actual == receipt["execution"], "Source actual execution replay differs")
            index = binding["assistant_call_index"]
            expected = deepcopy(actual["messages"])
            expected[0]["tool_calls"][0]["id"] = binding["native_call_id"]
            expected[1]["tool_call_id"] = binding["native_call_id"]
            ev.require(row["messages"][index:index + 2] == expected, "Source native call/result differs")
            ev.require(row["messages"][index + 2]["content"] == candidates.assistant_response(actual), "Source answer not result-derived")
            step = scenario["steps"][receipt["step_index"]]
            ev.require(not receipt["context"].get("confirmation") or (receipt["explicit_confirmation"] is True
                       and step["mode"] == "confirmation"), "No source explicit confirmation")
            identifier = row["id"] + ":step-" + str(receipt["step_index"])
            cases.append({"id": identifier, "messages": deepcopy(row["messages"][:index]),
                          "context": deepcopy(receipt["context"])})
            references.append({"id": identifier, "source_id": row["source_id"], "source_group_id": row["source_group_id"],
                "scenario_id": row["scenario_id"], "candidate_id": row["id"], "candidate_sha256": ev.digest(row),
                "registry_sha256": ev.world.sha256_json(group), "execution_id": receipt["id"],
                "execution_record_sha256": ev.world.sha256_json(receipt), "execution": actual,
                "family": row["metadata"]["family"], "prompt_form_family": row["metadata"]["prompt_form_family"],
                "step_mode": step["mode"], "source_manifest_sha256": ev.SOURCE_SHA})
    plan = {"version": ev.VERSION, "tools": ev.world.tools(), "cases": cases}
    ev.validate_plan(plan)
    return plan, references


def protected_content(source):
    from prepare_understanding_r7 import content_views
    return [{"id": row["id"] + ":prefix:" + str(i), "source_id": row["source_id"],
             "group_id": row["source_group_id"], "source_group_id": row["source_group_id"],
             "text": text, "training_eligible": False}
            for row in ev.read_rows(source / "candidates/dev.jsonl") for i, text in enumerate(content_views(row))]


def verify_code(folder):
    for name in CODE:
        ev.require(ev.sha(folder / "code" / name) == ev.sha(Path(__file__).parent / name),
                   "Use the evaluator code frozen in this package: " + name)


def prepare(source, tokenizer_path, out):
    import prepare_r7_tool_candidates as candidates
    from check_afrique_training_contract import load_tokenizer, ASSET_HASHES
    source, out = Path(source), Path(out)
    ev.require(not out.exists(), "Use a fresh evaluation folder")
    ev.require(candidates.verify_package(source)["manifest_sha256"] == ev.SOURCE_SHA, "Only original protected candidates allowed")
    plan, refs = derive_cases(source)
    tokenizer = load_tokenizer(Path(tokenizer_path))
    measurements = []
    for case in plan["cases"]:
        ids = ev.native_prompt(tokenizer, case["messages"], plan["tools"])
        measurements.append({"id": case["id"], "prompt_ids_sha256": ev.digest(ids), "prompt_tokens": len(ids)})
    out.mkdir(parents=True)
    for name in SOURCE_FILES:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    for name in CODE:
        target = out / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).parent / name, target)
    for name in ASSET_HASHES:
        target = out / "tokenizer" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(tokenizer_path) / name, target)
    ev.save(out / "plan.json", plan)
    ev.save(out / "references.json", refs)
    ev.save(out / "held-content.json", protected_content(source))
    ev.save(out / "token-measurements.json", measurements)
    protocol = {**ev.PROTOCOL, "tokenizer_files": ASSET_HASHES,
                "code_sha256": {name: ev.sha(out / "code" / name) for name in CODE},
                "tools_sha256": ev.digest(plan["tools"]), "plan_sha256": ev.digest(plan)}
    counts = {"cases": len(refs), "candidate_views": len({r["candidate_id"] for r in refs}),
              "scenarios": len({r["scenario_id"] for r in refs}),
              "source_groups": len({r["source_group_id"] for r in refs}),
              "families": dict(Counter(r["family"] for r in refs)), "maximum_generations": 2 * len(refs)}
    checksum = seal(out, {"version": ev.VERSION, "source_manifest_sha256": ev.SOURCE_SHA,
                         "protocol": protocol, "protocol_sha256": ev.digest(protocol), "counts": counts,
                         "training_eligible": False, "private_research_only": True})
    verify_evaluation(out, checksum)
    return {"manifest_sha256": checksum, **counts, "model_calls": 0}


def verify_evaluation(folder, checksum):
    folder = Path(folder)
    manifest = ev.verify_seal(folder, checksum)
    ev.require(manifest["version"] == ev.VERSION and manifest["source_manifest_sha256"] == ev.SOURCE_SHA,
               "Wrong evaluation version/source")
    verify_code(folder)
    ev.require(ev.sha(folder / "source/manifest.json") == ev.SOURCE_SHA, "Protected source pin differs")
    source_manifest = ev.read(folder / "source/manifest.json")
    for name in SOURCE_FILES:
        if name not in ("manifest.json", "PREPARED.json"):
            ev.require(ev.sha(folder / "source" / name) == source_manifest["artifacts"][name], "Protected source file differs")
    plan, refs = derive_cases(folder / "source")
    ev.require(ev.read(folder / "plan.json") == plan and ev.read(folder / "references.json") == refs,
               "Cases/reference identities differ from original protected source")
    ev.require(ev.read(folder / "held-content.json") == protected_content(folder / "source"), "Protection changed")
    ev.require(manifest["protocol_sha256"] == ev.digest(manifest["protocol"])
               and all(manifest["protocol"][k] == v for k, v in ev.PROTOCOL.items()), "Frozen protocol changed")
    ev.require(manifest["protocol"]["code_sha256"] == {name: ev.sha(folder / "code" / name) for name in CODE}
               and manifest["protocol"]["plan_sha256"] == ev.digest(plan)
               and manifest["protocol"]["tools_sha256"] == ev.digest(plan["tools"]), "Protocol code/input pins differ")
    return manifest, plan, refs


def audit_training(pack, checksum, evaluation):
    """Check consumed witnesses plus admitted snapshots, with original group aliases."""
    from prepare_understanding_r7 import protected_records
    pack = Path(pack)
    ev.require(ev.sha(pack / "manifest.json") == checksum, "Training pack differs from completed adapter")
    header = ev.read(pack / "manifest.json")
    if header["version"] == "understanding-adaptation-r6":
        import run_understanding_r6 as run
    elif header["version"] == "understanding-adaptation-r7":
        import run_understanding_r7 as run
    else:
        raise ValueError("Expected actual R6/R7 training pack")
    manifest, _, _ = run.verify_pack(pack)
    references = ev.read(pack / "source-manifests.json")
    snapshots = [ev.read(ev.safe_file(pack, r["manifest_path"])) for r in references]
    names = {"source-witnesses.jsonl"} | {r["records_path"] for r in snapshots}
    held = ev.read(Path(evaluation) / "held-content.json")
    scanned, all_rows = {}, []
    for name in sorted(names):
        rows = ev.read_rows(ev.safe_file(pack, name))
        all_rows.extend(rows)
        scanned[name] = {"records": len(rows), "sha256": ev.sha(pack / name)}
    blocked = protected_records(all_rows, held)
    ev.require(not blocked, "Held tool source/history leaked into training: " + json.dumps(list(blocked.items())[:3]))
    return {"training_manifest_sha256": checksum, "protected_units": len(held), "scanned": scanned,
            "overlap_free": True, "checks": ["connected_source_identity", "full_assistant_history_normalized_exact_and_8gram"],
            "scope": "available_admitted_snapshots_and_consumed_witnesses", "pretraining_overlap": "unknown"}


def verify_inference(bundle, checksum, variant):
    bundle = Path(bundle)
    manifest = ev.verify_seal(bundle, checksum)
    ev.require(manifest["version"] in ("r6-comparison-inference-v1", "r7-comparison-inference-v1"), "Unsupported bundle")
    name = "r7_inference" if variant == "r7_response" else "r6_inference"
    module = ev.import_file("_tool_eval_contract_" + checksum, bundle / "code" / (name + ".py"))
    ev.require(module.load_bundle(bundle, checksum) == manifest, "Inference training proof differs")
    return manifest


def stage(evaluation, evaluation_sha, bundle, bundle_sha, variant, training_pack, out, run_id):
    evaluation, bundle, out = Path(evaluation), Path(bundle), Path(out)
    ev.require(not out.exists(), "Use a fresh execution folder")
    ev.require(re.fullmatch(r"r7-tool-[a-z0-9-]{4,80}", run_id or ""), "Invalid run ID")
    manifest, plan, _ = verify_evaluation(evaluation, evaluation_sha)
    inference = verify_inference(bundle, bundle_sha, variant)
    ev.require(inference["decoding"] == ev.DECODING, "Inference decoding differs")
    for name, checksum in manifest["protocol"]["tokenizer_files"].items():
        ev.require(inference["files"]["tokenizer/" + name] == checksum, "Native tokenizer/template differs")
    training_sha = inference[variant.split("_")[0] + "_training"]["data_sha256"]
    audit = audit_training(training_pack, training_sha, evaluation)
    binding = {"version": ev.VERSION, "run_id": run_id, "evaluation_sha256": evaluation_sha,
               "plan_sha256": ev.digest(plan), "protocol_sha256": ev.digest({
                   "evaluation_protocol": manifest["protocol"], "runtime": inference["runtime"],
                   "model": inference["model"], "revision": inference["revision"], "base_sha256": inference["base_sha256"]}),
               "identity": ev.identity(inference, bundle_sha, variant), "training_manifest_sha256": training_sha,
               "training_audit_sha256": ev.digest(audit)}
    out.mkdir(parents=True)
    ev.save(out / "training-overlap.json", audit)
    ev.save(out / "binding.json", binding)
    checksum = seal(out, {"version": ev.VERSION, "evaluation": str(evaluation.resolve()),
                         "evaluation_sha256": evaluation_sha, "bundle": str(bundle.resolve()), "bundle_sha256": bundle_sha,
                         "variant": variant, "training_pack": str(Path(training_pack).resolve()),
                         "runtime": inference["runtime"], "launcher_sha256": ev.sha(__file__),
                         "gpu": "H100", "detached": True, "min_containers": 0, "max_containers": 1})
    return {"execution_sha256": checksum, "binding": binding}


def verify_execution(package, checksum):
    package = Path(package)
    manifest = ev.verify_seal(package, checksum)
    ev.require(manifest["version"] == ev.VERSION and manifest["launcher_sha256"] == ev.sha(__file__), "Execution launcher changed")
    evaluation, plan, refs = verify_evaluation(manifest["evaluation"], manifest["evaluation_sha256"])
    inference = verify_inference(manifest["bundle"], manifest["bundle_sha256"], manifest["variant"])
    binding = ev.read(package / "binding.json")
    ev.require(binding["identity"] == ev.identity(inference, manifest["bundle_sha256"], manifest["variant"])
               and binding["plan_sha256"] == ev.digest(plan) and binding["evaluation_sha256"] == manifest["evaluation_sha256"],
               "Execution binding differs")
    ev.require(binding["protocol_sha256"] == ev.digest({"evaluation_protocol": evaluation["protocol"],
        "runtime": inference["runtime"], "model": inference["model"], "revision": inference["revision"],
        "base_sha256": inference["base_sha256"]}), "Runtime protocol differs")
    audit = ev.read(package / "training-overlap.json")
    ev.require(binding["training_audit_sha256"] == ev.digest(audit) and audit["overlap_free"] is True
               and binding["training_manifest_sha256"] == audit["training_manifest_sha256"]
               == inference[manifest["variant"].split("_")[0] + "_training"]["data_sha256"], "Training overlap binding differs")
    return manifest, binding, plan, refs


def create_app(execution):
    import modal
    ev.require(sys.version_info[:2] == (3, 11) and sys.implementation.name == "cpython", "Use matching CPython 3.11 launcher")
    app = modal.App(APP)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    cache = modal.Volume.from_name(CACHE, create_if_missing=False)
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        *[name + "==" + version for name, version in execution["runtime"].items()]).env({
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    evaluation = Path(execution["evaluation"])
    code_sha = {name: ev.sha(evaluation / "code" / name) for name in CODE}
    image = image.add_local_dir(str(Path(execution["bundle"]).resolve()), "/bundle")
    image = image.add_local_dir(str(evaluation / "code"), "/worker")
    bundle_sha = execution["bundle_sha256"]

    @app.function(image=image, gpu="H100", cpu=4, memory=49152, timeout=7200, retries=0,
                  min_containers=0, max_containers=1, serialized=True,
                  volumes={"/cache": cache.read_only(), "/artifacts": volume})
    def remote(plan, binding):
        import hashlib
        import importlib.util
        import sys
        from pathlib import Path
        import modal
        for name, checksum in code_sha.items():
            if hashlib.sha256((Path("/worker") / name).read_bytes()).hexdigest() != checksum:
                raise ValueError("Remote evaluator source changed")
        sys.path.insert(0, "/worker")
        path = Path("/worker/r7_tool_eval.py")
        spec = importlib.util.spec_from_file_location("_r7_tool_eval_worker", path)
        worker = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = worker
        spec.loader.exec_module(worker)
        worker.require(worker.validate_plan(plan) == binding["plan_sha256"], "Remote plan differs")
        worker.require(binding["identity"]["manifest_sha256"] == bundle_sha, "Remote bundle binding differs")
        claims = modal.Dict.from_name("ghana-r7-tool-eval-invocations", create_if_missing=True)
        owner = modal.current_function_call_id()
        claim = {"call_id": owner, "binding": binding}
        claims.put(binding["run_id"], claim, skip_if_exists=True)
        worker.require(claims.get(binding["run_id"]) == claim, "Another invocation owns this run")
        volume.reload()
        root = Path("/artifacts/understanding-adaptation-r7/tool-eval") / binding["run_id"]
        return worker.evaluate(plan, binding,
            lambda: worker.NativeRuntime("/bundle", bundle_sha, binding["identity"]["variant"]),
            root / "progress.json", commit=volume.commit)

    return app, remote


@contextmanager
def receipt_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    ev.require(not lock.is_symlink(), "Linked receipt lock")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def launch(package, checksum, receipt_path, profile, environment, authorize_gpu=False):
    ev.require(authorize_gpu, "Explicit --authorize-gpu required; no GPU called")
    execution, binding, plan, _ = verify_execution(package, checksum)
    ev.require(sys.version_info[:2] == (3, 11), "Use the existing Python 3.11 launcher environment")
    ev.require(not Path(receipt_path).resolve().is_relative_to(Path(package).resolve()), "Receipt outside sealed package required")
    ev.require(audit_training(execution["training_pack"], binding["training_manifest_sha256"], execution["evaluation"])
               == ev.read(Path(package) / "training-overlap.json"), "Launch-time training audit differs")
    with receipt_lock(receipt_path):
        ev.require(not Path(receipt_path).exists(), "Existing receipt; collect, do not duplicate or auto-resubmit")
        os.environ["MODAL_PROFILE"], os.environ["MODAL_ENVIRONMENT"] = profile, environment
        import modal
        claims = modal.Dict.from_name(CLAIMS, create_if_missing=True)
        ev.require(claims.get(binding["run_id"]) is None, "Run already claimed; recover original receipt")
        receipt = {"version": ev.VERSION, "execution_sha256": checksum, "binding": binding,
                   "profile": profile, "environment": environment, "state": "preparing", "call_id": None}
        ev.save(receipt_path, receipt)
        try:
            app, function = create_app(execution)
            with modal.enable_output(), app.run(detach=True):
                receipt["state"] = "submission_unknown"
                ev.save(receipt_path, receipt)
                call = function.spawn(plan, binding)
                receipt.update(state="submitted", call_id=call.object_id, app_id=app.app_id)
                ev.save(receipt_path, receipt)
                claim = {"call_id": call.object_id, "binding": binding}
                claims.put(binding["run_id"], claim, skip_if_exists=True)
                ev.require(claims.get(binding["run_id"]) == claim, "Invocation claim differs")
        except Exception as error:
            receipt["error"] = type(error).__name__ + ": " + str(error)
            ev.save(receipt_path, receipt)
            raise
        return receipt


def score(package, checksum, progress, reviews=None):
    from check_afrique_training_contract import load_tokenizer
    from prepare_r7_tool_candidates import assistant_response
    execution, binding, plan, refs = verify_execution(package, checksum)
    ev.require(progress["binding"] == binding, "Progress run identity differs")
    ev.require(progress["snapshot_sha256"] == ev.digest({k: v for k, v in progress.items() if k != "snapshot_sha256"}),
               "Progress hash differs")
    tokenizer = load_tokenizer(Path(execution["evaluation"]) / "tokenizer")
    cases, refs = ({r["id"]: r for r in items} for items in (plan["cases"], refs))
    records = progress["results"]
    ev.require(len({r["id"] for r in records}) == len(records) and all(r["id"] in cases for r in records), "Unknown/duplicate output")
    ev.require(progress["state"] != "complete" or len(records) == len(cases), "Premature complete state")
    reviews = reviews or []
    ev.require(len({r["id"] for r in reviews}) == len(reviews) and all(r["id"] in {x["id"] for x in records} for r in reviews),
               "Duplicate/unknown review")
    by_review = {r["id"]: r for r in reviews}
    scores, queue = [], []
    for record in records:
        case = cases[record["id"]]
        ev.verify_record(record, case, tokenizer, plan["tools"], binding["identity"])
        scored = ev.score_case(record, refs[record["id"]], assistant_response, by_review.get(record["id"]))
        scores.append(scored)
        if scored["grounded_final"] == "needs_review":
            queue.append({"id": record["id"], "record_sha256": ev.digest(record),
                "result_sha256": record["execution"]["result_sha256"], "messages": case["messages"],
                "call": record["call"], "execution": record["execution"], "final_answer": record["final_answer"],
                "scope": "grounding in actual execution, including simulated/empty/error boundaries; not whole translation gold"})
    return {"version": ev.VERSION, "verified": True, "binding": binding,
            "evidence": {"package": str(Path(package).resolve()), "execution_sha256": checksum,
                         "progress": progress, "reviews": reviews},
            "progress_sha256": ev.digest(progress), "reviews_sha256": ev.digest(reviews),
            "summary": ev.summarize(scores, len(cases)), "scores": scores, "review_queue": queue}


def replay_report(path):
    report = ev.read(path)
    source = report["evidence"]
    replay = score(source["package"], source["execution_sha256"], source["progress"], source["reviews"])
    ev.require(replay == report, "Comparison report differs from replayed actual outputs")
    return replay


def collect(package, receipt_path, out):
    receipt = ev.read(receipt_path)
    execution, binding, _, _ = verify_execution(package, receipt["execution_sha256"])
    ev.require(receipt["binding"] == binding, "Receipt binding differs")
    out = Path(out)
    ev.require(not out.exists(), "Use fresh collection directory")
    os.environ["MODAL_PROFILE"], os.environ["MODAL_ENVIRONMENT"] = receipt["profile"], receipt["environment"]
    import modal
    claim = modal.Dict.from_name(CLAIMS, create_if_missing=False).get(binding["run_id"])
    if claim:
        ev.require(claim["binding"] == binding and receipt["call_id"] in (None, claim["call_id"]), "Remote claim differs")
        receipt.update(call_id=claim["call_id"], state="submitted")
        ev.save(receipt_path, receipt)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    try:
        progress = ev.parse_json(b"".join(volume.read_file("/" + PREFIX + "/" + binding["run_id"] + "/progress.json")))
    except FileNotFoundError:
        return {"state": "pending", "call_id": receipt["call_id"]}
    report = score(package, receipt["execution_sha256"], progress)
    out.mkdir(parents=True)
    ev.save(out / "progress.json", progress)
    ev.save(out / "report.json", report)
    ev.save(out / "review-queue.json", report["review_queue"])
    return report["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_p = sub.add_parser("prepare")
    prepare_p.add_argument("--source", type=Path, required=True)
    prepare_p.add_argument("--tokenizer", type=Path, required=True)
    prepare_p.add_argument("--out", type=Path, required=True)
    stage_p = sub.add_parser("stage")
    for name in ("evaluation", "evaluation-sha", "bundle", "bundle-sha", "training-pack", "out", "run-id"):
        stage_p.add_argument("--" + name, required=True)
    stage_p.add_argument("--variant", choices=("r6_response", "r7_response"), required=True)
    launch_p = sub.add_parser("launch")
    for name in ("package", "execution-sha", "receipt", "profile", "environment"):
        launch_p.add_argument("--" + name, required=True)
    launch_p.add_argument("--authorize-gpu", action="store_true")
    collect_p = sub.add_parser("collect")
    for name in ("package", "receipt", "out"):
        collect_p.add_argument("--" + name, required=True)
    score_p = sub.add_parser("score")
    for name in ("package", "execution-sha", "progress", "out"):
        score_p.add_argument("--" + name, required=True)
    score_p.add_argument("--reviews")
    compare_p = sub.add_parser("compare")
    for name in ("r6", "r7", "out"):
        compare_p.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.tokenizer, args.out)
    elif args.command == "stage":
        result = stage(args.evaluation, args.evaluation_sha, args.bundle, args.bundle_sha,
                       args.variant, args.training_pack, args.out, args.run_id)
    elif args.command == "launch":
        result = launch(args.package, args.execution_sha, args.receipt, args.profile, args.environment, args.authorize_gpu)
    elif args.command == "collect":
        result = collect(args.package, args.receipt, args.out)
    elif args.command == "score":
        result = score(args.package, args.execution_sha, ev.read(args.progress), ev.read(args.reviews) if args.reviews else None)
        ev.require(not Path(args.out).exists(), "Use fresh score output")
        ev.save(args.out, result)
    else:
        result = ev.compare_reports(replay_report(args.r6), replay_report(args.r7))
        ev.require(not Path(args.out).exists(), "Use fresh comparison output")
        ev.save(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
