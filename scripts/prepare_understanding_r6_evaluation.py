#!/usr/bin/env python3
"""Offline, lossless adapter for an owner-supplied private R5 fresh proposal.

No prompt authoring, model calls, overlap clearance or execution authorization.
The source design remains a proposal; freezing transport bytes is not approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

VERSION = "r5-fresh-evaluation-adapter-v1"
SOURCE_VERSION = "r5-fresh-evaluation-proposal-v1"
KIND = "fresh_proposal_not_blind_final"
DECODING = {"do_sample": False, "enable_thinking": False, "max_new_tokens": 256,
            "num_beams": 1, "repetition_penalty": 1.0, "seed": 17, "truncate": False}
VARIANTS = ["base", "r4_response", "r5_response"]
ARTIFACTS = ("source.json", "validator.py", "heldout-seal.json", "design-audit.json",
             "cases.json", "rollouts.json", "language-loss.json")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "Nonfinite JSON"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def text(value):
    require(isinstance(value, str) and value.strip() and "\ufffd" not in value
            and not any(0xD800 <= ord(c) <= 0xDFFF for c in value), "Invalid proposal text")


def messages(value):
    require(isinstance(value, list) and value, "Missing fixed messages")
    start = int(value[0].get("role") == "system")
    require((len(value) - start) % 2 == 1, "Fixed context must end with user")
    for i, message in enumerate(value):
        role = "system" if start and i == 0 else "user" if (i - start) % 2 == 0 else "assistant"
        require(set(message) == {"role", "content"} and message["role"] == role, "Invalid fixed roles")
        text(message["content"])


def schema(path=None):
    import importlib.util
    path = Path(path) if path is not None else Path(__file__).with_name("test_r5_fresh_evaluation.py")
    spec = importlib.util.spec_from_file_location("_r5_fresh_schema", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def adapt(source, validator=None):
    require(source["version"] == SOURCE_VERSION and source["status"] == "private_proposal_not_sealed_final_test",
            "Expected an unexecuted private proposal, not approved final-test evidence")
    comparison = source["comparison"]
    require(comparison["variants"] == ["base", "r4", "r5"], "Wrong proposed checkpoint set")
    proposal = comparison["decoding_proposal"]
    require(set(proposal) in (set(DECODING), set(DECODING) - {"enable_thinking"})
            and all(proposal[k] == DECODING[k] and type(proposal[k]) is type(DECODING[k]) for k in proposal),
            "Proposal decoding differs from the pinned comparison")
    materialized = (validator or schema()).validate(source)
    require(len(materialized) <= 30000, "Proposal exceeds finite case bound")
    cases, rollouts = [], []
    for case in materialized:
        inputs = case["stimulus"]
        mode = case["mode"]
        if mode == "fixed_context":
            messages(inputs["messages"])
        else:
            for user in inputs["user_turns"]:
                text(user)
        item = {"id": case["id"], "source_group_id": case["source_group_id"],
                "cohort": "fresh_" + case["language"], "input_language": case["language"],
                "synthetic_informal3": case["synthetic_informal3"], "origin": case["origin"],
                "source_case_sha256": case["case_sha256"], "source_input_sha256": case["input_sha256"],
                "phenomena": case["phenomena"], "review_slots": case["reviews"], **inputs}
        (cases if mode == "fixed_context" else rollouts).append(item)
    return cases, rollouts

def source_proof(folder, source, validator):
    seal, audit = read(folder / "heldout-seal.json"), read(folder / "design-audit.json")
    require(seal["version"] == "r5-private-heldout-seal-v1" and seal["private"] is True
            and all(seal[k] is False for k in ("training_eligible", "execution_authorized", "native_certification", "clinical_certification"))
            and type(seal["model_calls"]) is int and seal["model_calls"] == 0,
            "Held-out proposal seal boundary differs")
    require(seal["artifacts"] == {"r5_fresh_evaluation.json": sha(folder / "source.json"),
                                  "audit.json": sha(folder / "design-audit.json")}
            and seal["validator"]["sha256"] == sha(folder / "validator.py")
            and audit["corpus_file_sha256"] == sha(folder / "source.json")
            and audit["validator_file_sha256"] == sha(folder / "validator.py"), "Held-out design/audit identity differs")
    require(audit["private"] is True and audit["execution_authorized"] is False
            and type(audit["model_calls"]) is int and audit["model_calls"] == 0, "Design audit authorization differs")
    measured = validator.summary(source, validator.validate(source))
    require(seal["inventory_sha256"] == measured["inventory_sha256"]
            and all(audit[k] == measured[k] for k in ("inventory_sha256", "counts", "groups", "cases")),
            "Held-out audit inventory differs")
    return {"seal_sha256": sha(folder / "heldout-seal.json"), "audit_sha256": sha(folder / "design-audit.json"),
            "validator_sha256": sha(folder / "validator.py"), "inventory_sha256": measured["inventory_sha256"]}


def descriptor(source, source_sha, cases, rollouts, proof):
    return {"version": VERSION, "kind": KIND, "private": True, "training_eligible": False,
            "execution_authorized": False, "variants": VARIANTS, "decoding": DECODING,
            "source": {"version": source["version"], "status": source["status"], "sha256": source_sha, **proof},
            "counts": {"fixed_prompts": len(cases), "rollout_scenarios": len(rollouts),
                       "rollout_user_turns": sum(len(r["user_turns"]) for r in rollouts),
                       "source_groups": len({g["source_group_id"] for g in source["groups"]}), "language_rows": 0},
            "source_audit": "Required separately against actual sealed training inputs before execution preparation"}


def load(folder, expected_sha):
    folder = Path(folder).resolve(strict=True)
    require(isinstance(expected_sha, str) and re.fullmatch(r"[a-f0-9]{64}", expected_sha)
            and sha(folder / "manifest.json") == expected_sha
            and read(folder / "FROZEN.json") == {"manifest_sha256": expected_sha}, "Fresh suite identity differs")
    require(not any(p.is_symlink() for p in folder.rglob("*")), "Linked fresh suite artifact")
    require({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            == set(ARTIFACTS) | {"manifest.json", "FROZEN.json"}, "Unexpected fresh suite artifact")
    manifest = read(folder / "manifest.json")
    require(set(manifest["artifacts"]) == set(ARTIFACTS)
            and all(sha(folder / name) == checksum for name, checksum in manifest["artifacts"].items()), "Fresh suite artifact changed")
    source = read(folder / "source.json")
    validator = schema(folder / "validator.py")
    cases, rollouts = adapt(source, validator)
    proof = source_proof(folder, source, validator)
    require(manifest == {**descriptor(source, sha(folder / "source.json"), cases, rollouts, proof), "artifacts": manifest["artifacts"]},
            "Fresh suite metadata differs")
    require(read(folder / "cases.json") == cases and read(folder / "rollouts.json") == rollouts,
            "Fresh mapping differs from exact source inputs/reviews")
    require(read(folder / "language-loss.json") == {"enabled": False, "rows": 0}, "Fresh language loss is not bound")
    return manifest, cases, rollouts, []


def prepare(source, source_sha, out, *, seal, seal_sha):
    source, out = Path(source), Path(out)
    require(not source.is_symlink() and sha(source) == source_sha, "Fresh source identity differs")
    seal = Path(seal)
    require(not seal.is_symlink() and sha(seal) == seal_sha, "Held-out seal identity differs")
    audit = seal.parent / "audit.json"
    require(not audit.is_symlink(), "Linked design audit")
    require(not out.exists() and not out.is_symlink(), "Use a fresh immutable suite directory")
    require(not any(out.resolve().is_relative_to(root.resolve()) for root in (source.parent, seal.parent)),
            "Suite output must be outside the private source design")
    original = read(source)
    cases, rollouts = adapt(original)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".r5-fresh-evaluation-") as temporary:
        staged = Path(temporary) / "suite"
        staged.mkdir(mode=0o700)
        shutil.copyfile(source, staged / "source.json")
        shutil.copyfile(Path(__file__).with_name("test_r5_fresh_evaluation.py"), staged / "validator.py")
        shutil.copyfile(seal, staged / "heldout-seal.json")
        shutil.copyfile(audit, staged / "design-audit.json")
        require(sha(staged / "heldout-seal.json") == seal_sha, "Held-out seal changed during preparation")
        require(sha(staged / "source.json") == source_sha, "Fresh source changed during preparation")
        save(staged / "cases.json", cases)
        save(staged / "rollouts.json", rollouts)
        save(staged / "language-loss.json", {"enabled": False, "rows": 0})
        proof = source_proof(staged, original, schema(staged / "validator.py"))
        manifest = {**descriptor(original, source_sha, cases, rollouts, proof),
                    "artifacts": {name: sha(staged / name) for name in ARTIFACTS}}
        save(staged / "manifest.json", manifest)
        checksum = sha(staged / "manifest.json")
        save(staged / "FROZEN.json", {"manifest_sha256": checksum})
        for artifact in staged.iterdir():
            artifact.chmod(0o600)
        load(staged, checksum)
        staged.rename(out)
    return {"evaluation": str(out.resolve()), "evaluation_sha256": checksum, "counts": manifest["counts"],
            "kind": KIND, "execution_authorized": False, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--source-sha", required=True)
    prep.add_argument("--heldout-seal", type=Path, required=True)
    prep.add_argument("--heldout-seal-sha", required=True)
    prep.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--evaluation", type=Path, required=True)
    verify.add_argument("--evaluation-sha", required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.source_sha, args.out, seal=args.heldout_seal, seal_sha=args.heldout_seal_sha) if args.action == "prepare" else {
        key: value for key, value in load(args.evaluation, args.evaluation_sha)[0].items()
        if key in {"kind", "counts", "source", "execution_authorized"}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
