#!/usr/bin/env python3
"""Offline official IFEval scoring of receipt-bound outputs, never model execution."""
import argparse
import base64
from collections import Counter
from contextlib import contextmanager
import copy
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time

import run_understanding_r5_ifeval as runner
from run_understanding_r5_ifeval import read, save, sha, digest, require

VERSION = "r5-official-ifeval-score-v1"
MODES = ("strict", "loose")
CASE_SECONDS = 10
SEED = 17


@contextmanager
def bounded_scoring():
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Existing CPU deadline; use a separate scoring process")
    def timeout(signum, frame):
        raise TimeoutError("Official scorer exceeded the per-mode CPU wall deadline")
    old = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, CASE_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class OfficialScorer:
    def __init__(self, folder, source_sha):
        self.folder = Path(folder).resolve()
        runner.source.load(self.folder, source_sha)
        versions = {name: importlib.metadata.version(name) for name in runner.source.SCORER_ENV}
        require(versions == runner.source.SCORER_ENV, "Pinned scorer dependency versions differ")
        require(os.environ.get("PYTHONHASHSEED") == "0", "Start scorer with PYTHONHASHSEED=0")
        records = {}
        for name in versions:
            distribution = importlib.metadata.distribution(name)
            record = distribution.read_text("RECORD")
            require(bool(record), "Missing installed dependency file receipt")
            for item in distribution.files or []:
                if item.hash is not None:
                    with Path(distribution.locate_file(item)).open("rb") as stream:
                        actual = hashlib.file_digest(stream, item.hash.mode).digest()
                    require(base64.urlsafe_b64encode(actual).rstrip(b"=").decode() == item.hash.value,
                            "Installed scorer dependency bytes differ: " + name)
            records[name] = hashlib.sha256(record.encode()).hexdigest()
        upstream = self.folder / "upstream"
        for name, module in list(sys.modules.items()):
            if name.startswith("instruction_following_eval"):
                path = getattr(module, "__file__", None)
                paths = [Path(path).resolve()] if path is not None else [Path(p).resolve() for p in getattr(module, "__path__", [])]
                require(paths and all(p.is_relative_to(upstream) for p in paths), "Different official scorer already imported")
        sys.path.insert(0, str(upstream))
        prior = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            self.lib = importlib.import_module("instruction_following_eval.evaluation_lib")
            import nltk
            import langdetect
            nltk.data.path[:] = [str(self.folder / "nltk_data")]
            self.detectors = langdetect.DetectorFactory
            self.detectors.seed = SEED
            # Fail before scoring if packaged sentence resources cannot be loaded.
            importlib.import_module("instruction_following_eval.instructions_util").count_sentences("One. Another sentence.")
        finally:
            sys.dont_write_bytecode = prior
        self.proof = {"version": VERSION, "source_sha256": source_sha, "source_revision": runner.source.REVISION,
            "scorer_sha256": runner.source.UPSTREAM["instruction_following_eval/evaluation_lib.py"],
            "dependencies": versions, "installed_distribution_records_sha256": records, "dependency_file_hashes_verified": True,
            "python": platform.python_version(),
            "seed_per_case_per_mode": SEED, "pythonhashseed": "0", "cpu_seconds_per_mode": CASE_SECONDS,
            "resources": runner.source.NLTK_FILES, "scorer_modified": False, "network_required": False}

    def score(self, row, response, mode):
        require(mode in MODES and isinstance(response, str), "Invalid scoring input")
        random.seed(SEED)
        self.detectors.seed = SEED
        inp = self.lib.InputExample(**copy.deepcopy(row))
        function = getattr(self.lib, "test_instruction_following_" + mode)
        with bounded_scoring():
            value = dataclasses.asdict(function(inp, {row["prompt"]: response}))
        require(value["prompt"] == row["prompt"] and value["response"] == response
                and value["instruction_id_list"] == row["instruction_id_list"]
                and len(value["follow_instruction_list"]) == len(row["instruction_id_list"])
                and all(type(v) is bool for v in value["follow_instruction_list"])
                and type(value["follow_all_instructions"]) is bool
                and value["follow_all_instructions"] == all(value["follow_instruction_list"]), "Official scorer output differs")
        return value


def score_slots(rows, generation_journal, scores, scorer, deadline, max_records=1623, clock=time.monotonic):
    expected = {digest(s["key"]) + ".json" for s in runner.slots(rows)}
    require({p.name for p in (scores.folder / "records").glob("*.json")} <= expected, "Unexpected scoring records")
    count = 0
    for slot in runner.slots(rows):
        record = generation_journal.get(slot["key"])
        input_sha = digest({"source_case": slot["case"], "generation_record": record})
        previous = scores.get(slot["key"])
        if previous is not None:
            require(previous["inputs_sha256"] == input_sha, "Changed generation response during scoring resume")
            continue
        if count >= max_records or clock() + 2 * CASE_SECONDS + 1 >= deadline:
            return "paused"
        value = {"case_key": slot["case_key"], "variant": slot["variant"], "inputs_sha256": input_sha,
                 "generation_state": (record or {}).get("state", "pending"), "official": {}, "scorer_errors": {},
                 "generation_record_sha256": digest(record), "source_case_sha256": digest(slot["case"]),
                 "cap_hit": (record or {}).get("result", {}).get("cap_hit", False),
                 "finish_reason": (record or {}).get("result", {}).get("finish_reason")}
        started = list((scores.folder / "attempts").glob("*/" + digest(slot["key"]) + ".json"))
        if started:
            require(all(read(p)["id"] == slot["key"] and read(p)["inputs_sha256"] == input_sha for p in started), "Interrupted scoring input differs")
            value["state"] = "scoring_indeterminate"
        elif record and record.get("verified") is True:
            scores.begin(slot["key"], 1, input_sha)
            response = record["result"]["output"]
            value["response_sha256"] = digest(response)
            for mode in MODES:
                try:
                    value["official"][mode] = scorer.score(slot["case"], response, mode)
                except Exception as error:
                    value["scorer_errors"][mode] = {"error_type": type(error).__name__, "error": str(error)[:1000]}
            value["state"] = "scored" if not value["scorer_errors"] else "scoring_error"
        else:
            value["state"] = "generation_" + value["generation_state"]
        scores.finish(slot["key"], value)
        count += 1
    return "finalized"


def metric(rows, values, mode, instruction_filter=lambda _: True):
    chosen = [(row, [i for i, name in enumerate(row["instruction_id_list"]) if instruction_filter(name)]) for row in rows]
    chosen = [(row, indices) for row, indices in chosen if indices]
    expected_prompts, expected_instructions = len(chosen), sum(len(indices) for _, indices in chosen)
    prompt_pass = instruction_pass = scored_prompts = scored_instructions = 0
    for row, indices in chosen:
        output = (values.get(row["key"]) or {}).get("official", {}).get(mode)
        if output is not None:
            scored_prompts += 1
            scored_instructions += len(indices)
            prompt_pass += output["follow_all_instructions"]
            instruction_pass += sum(output["follow_instruction_list"][i] for i in indices)
    complete = expected_prompts > 0 and expected_prompts == scored_prompts
    return {"expected_prompts": expected_prompts, "expected_instructions": expected_instructions,
            "scored_prompts": scored_prompts, "scored_instructions": scored_instructions,
            "unscored_prompts": expected_prompts - scored_prompts, "unscored_instructions": expected_instructions - scored_instructions,
            "passed_prompts": prompt_pass, "passed_instructions": instruction_pass, "complete_coverage": complete,
            "prompt_accuracy": prompt_pass / expected_prompts if complete else None,
            "instruction_accuracy": instruction_pass / expected_instructions if complete else None,
            "observed_prompt_fraction": prompt_pass / scored_prompts if scored_prompts else None,
            "observed_instruction_fraction": instruction_pass / scored_instructions if scored_instructions else None,
            "full_denominator_prompt_lower_bound": prompt_pass / expected_prompts if expected_prompts else None,
            "full_denominator_instruction_lower_bound": instruction_pass / expected_instructions if expected_instructions else None}


def report(out, rows, scores, state):
    models, cases = {}, []
    types = sorted({i for row in rows for i in row["instruction_id_list"]})
    categories = sorted({i.split(":", 1)[0] for i in types})
    for variant in runner.VARIANTS:
        values = {row["key"]: scores.get(str(row["key"]) + ":" + variant) for row in rows}
        counts = Counter(v["state"] if v else "scoring_pending" for v in values.values())
        models[variant] = {"states": dict(counts), "generation_states": dict(Counter(v["generation_state"] for v in values.values() if v)),
            "cap_hits": sum(v.get("cap_hit", False) for v in values.values() if v),
            "overall": {mode: metric(rows, values, mode) for mode in MODES},
            "by_instruction_type": {name: {mode: metric(rows, values, mode, lambda i, n=name: i == n) for mode in MODES} for name in types},
            "by_category": {name: {mode: metric(rows, values, mode, lambda i, n=name: i.split(":", 1)[0] == n) for mode in MODES} for name in categories}}
        for row in rows:
            value = values[row["key"]]
            cases.append(value or {"case_key": row["key"], "variant": variant, "state": "scoring_pending"})
        for mode in MODES:
            outputs = [v["official"][mode] for v in values.values() if v and mode in v["official"]]
            (Path(out) / ("scored-outputs-" + variant + "-" + mode + ".jsonl")).write_text(
                "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in outputs))
    result = {"version": VERSION, "binding": scores.binding, "state": state, "models": models,
              "full_suite_scoring_complete": all(m["overall"][mode]["complete_coverage"] for m in models.values() for mode in MODES),
              "subgroup_prompt_metric": "all instructions on prompts containing the named type/category",
              "partial_fractions_are_not_full_benchmark_scores": True, "completion_is_accuracy": False,
              "twi_ability_established": False, "leaderboard_comparability_claimed": False}
    save(Path(out) / "scores.json", result)
    save(Path(out) / "score-cases.json", {"binding": scores.binding, "cases": cases})
    save(Path(out) / "score-manifest.json", {"binding": scores.binding,
        "files": {n: h for n, h in runner.files(out).items() if n != "score-manifest.json"}})
    runner.source.private(out)
    return result


def score(package, checksum, results, out, *, resume=False, seconds=1800, max_records=1623):
    require(type(seconds) is int and 30 <= seconds <= 7200 and type(max_records) is int and 1 <= max_records <= 1623, "Invalid CPU scoring bounds")
    out = Path(out).absolute()
    require((resume and out.is_dir()) or (not resume and not out.exists()), "Use a new score directory or explicit --resume")
    require(not out.is_symlink() and not any(out.resolve().is_relative_to(Path(p).resolve()) for p in (package, results)), "Score output inside immutable evidence")
    execution, rows, generation_journal = runner.audit_results(Path(package), checksum, results)
    for module in (sys.modules[__name__], runner, runner.source, runner.generation, runner.common):
        path = Path(module.__file__)
        require(sha(path) == execution["files"]["code/" + path.name], "Scoring wrapper differs from execution snapshot")
    scorer = OfficialScorer(Path(package) / "source", execution["source_sha256"])
    bound = {"version": VERSION, "generation": generation_journal.binding,
             "results_manifest_sha256": sha(Path(results) / "results-manifest.json"), "scorer": scorer.proof}
    core, _ = runner.common.activate(package)
    out.mkdir(parents=True, exist_ok=resume, mode=0o700)
    scores = core.Journal(out / "journal", bound)
    state = score_slots(rows, generation_journal, scores, scorer, time.monotonic() + seconds, max_records)
    runner.source.load(Path(package) / "source", execution["source_sha256"])
    return report(out, rows, scores, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("execution", "results", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--max-records", type=int, default=1623)
    args = parser.parse_args()
    value = score(args.execution, args.execution_sha, args.results, args.out, resume=args.resume, seconds=args.seconds, max_records=args.max_records)
    print(json.dumps({"out": str(args.out), "state": value["state"], "full_suite_scoring_complete": value["full_suite_scoring_complete"]}, indent=2))


if __name__ == "__main__":
    main()
