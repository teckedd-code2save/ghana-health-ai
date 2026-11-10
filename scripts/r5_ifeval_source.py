#!/usr/bin/env python3
"""Fetch/verify the full official IFEval revision and offline scorer resources."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from run_understanding_r5_evaluation import digest, files, read, require, save, sha

VERSION = "r5-official-ifeval-source-v1"
REVISION = "0eebfd403ba438bfd14cb70b9b24a7745c997083"
URL = "https://raw.githubusercontent.com/google-research/google-research/" + REVISION + "/"
UPSTREAM = {
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "instruction_following_eval/README.md": "fef3c44408e4b39837af345095ccce7da798853fd60ea94a1de63afa9d3d67b3",
    "instruction_following_eval/data/input_data.jsonl": "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
    "instruction_following_eval/evaluation_lib.py": "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    "instruction_following_eval/evaluation_main.py": "e6df07a04d25a0e7134933ba2400c26e0129a243e449c88df55043e583fc4b4a",
    "instruction_following_eval/instructions.py": "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    "instruction_following_eval/instructions_registry.py": "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    "instruction_following_eval/instructions_util.py": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
    "instruction_following_eval/requirements.txt": "1b1716c9ac21b9cc2bd0d7354ed8f7b6982ccf35598529601a7c08508acec94a",
    "instruction_following_eval/instructions_test.py": "e677c190c393162f7733e89ca3b47ae98b6d9e816d800681d6e14ad918128de5",
    "instruction_following_eval/instructions_util_test.py": "e8e17d434e074c6e2ad997334f4d743b3cafe0abf15cca46d4cf97b94d67ccfe",
}
DATA = "instruction_following_eval/data/input_data.jsonl"
NLTK_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_URL = "https://raw.githubusercontent.com/nltk/nltk_data/" + NLTK_REVISION + "/packages/tokenizers/punkt_tab.zip"
NLTK_SHA = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
NLTK_FILES = {
    "punkt_tab/english/collocations.tab": "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf",
    "punkt_tab/english/sent_starters.txt": "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac",
    "punkt_tab/english/abbrev_types.txt": "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779",
    "punkt_tab/english/ortho_context.tab": "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c",
    "punkt_tab/README": "d3251cae66a9359bd68c039e3a46172a05ca9df0b27dcdfa23ae594d68d27ec4",
}
SCORER_ENV = {"absl-py": "2.3.1", "langdetect": "1.0.9", "nltk": "3.9.2", "immutabledict": "4.2.2",
              "click": "8.1.8", "joblib": "1.5.3", "regex": "2026.5.9", "tqdm": "4.67.1", "six": "1.17.0"}


def private(folder):
    folder = Path(folder)
    folder.chmod(0o700)
    for path in folder.rglob("*"):
        require(not path.is_symlink(), "Linked private artifact")
        path.chmod(0o700 if path.is_dir() else 0o600)


def inventory(rows):
    require(len(rows) == len({r["key"] for r in rows}) == len({r["prompt"] for r in rows}) == 541,
            "Full IFEval requires all 541 unique prompts/keys")
    for row in rows:
        require(set(row) == {"key", "prompt", "instruction_id_list", "kwargs"}
                and type(row["key"]) is int and isinstance(row["prompt"], str) and row["prompt"].strip()
                and len(row["instruction_id_list"]) == len(row["kwargs"]) > 0, "Invalid official row")
    instructions = [i for row in rows for i in row["instruction_id_list"]]
    require(len(instructions) == 834 and len(set(instructions)) == 25, "Official instruction coverage differs")
    constraints = [{"key": row["key"], "instruction_id": i, **kwargs}
                   for row in rows for i, kwargs in zip(row["instruction_id_list"], row["kwargs"], strict=True)
                   if i.startswith("length_constraints:")]
    bounds = {name: max(c.get(name, 0) or 0 for c in constraints)
              for name in ("num_words", "num_sentences", "num_paragraphs")}
    minimum_words = max(c["num_words"] for c in constraints if "num_words" in c and c["relation"] == "at least")
    require(bounds["num_words"] == minimum_words == 1200, "Word-budget justification changed")
    return {"prompts": 541, "instruction_instances": 834, "instruction_types": sorted(set(instructions)),
            "maximum_length_constraints": bounds, "maximum_minimum_words": minimum_words,
            "length_constraints": constraints, "inventory_sha256": digest(rows)}


def descriptor(rows):
    return {"version": VERSION, "revision": REVISION, "source_url": URL,
            "license": "Apache-2.0; upstream LICENSE and source headers preserved",
            "dataset_sha256": UPSTREAM[DATA], "scope": "full_official_english_prompt_ifeval_not_twi_ability",
            "training_eligible": False, "private": True, "model_calls": 0,
            "scorer_environment": SCORER_ENV, "nltk_resource": {"url": NLTK_URL, "sha256": NLTK_SHA},
            "inventory": inventory(rows), "files": {
                **{"upstream/" + n: h for n, h in UPSTREAM.items()}, "resources/punkt_tab.zip": NLTK_SHA,
                **{"nltk_data/tokenizers/" + n: h for n, h in NLTK_FILES.items()}}}


def load(folder, expected_sha=None):
    folder = Path(folder)
    checksum = sha(folder / "manifest.json")
    require(expected_sha is None or checksum == expected_sha, "IFEval source manifest differs")
    require(read(folder / "SEALED.json") == {"manifest_sha256": checksum}, "IFEval source seal differs")
    manifest = read(folder / "manifest.json")
    expected = {**{"upstream/" + n: h for n, h in UPSTREAM.items()}, "resources/punkt_tab.zip": NLTK_SHA,
                **{"nltk_data/tokenizers/" + n: h for n, h in NLTK_FILES.items()},
                "manifest.json": checksum, "SEALED.json": sha(folder / "SEALED.json")}
    require(files(folder) == expected, "Official source/resource bytes or inventory differ")
    rows = [json.loads(line) for line in (folder / "upstream" / DATA).read_text().splitlines()]
    require(manifest == descriptor(rows), "IFEval source description differs")
    return manifest, rows


def fetch(out):
    out = Path(out).absolute()
    require(not out.exists() and not out.is_symlink(), "Use a new private source directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".ifeval-source-") as temp:
        stage = Path(temp) / "source"
        stage.mkdir(mode=0o700)
        for name, checksum in UPSTREAM.items():
            with urllib.request.urlopen(URL + name, timeout=60) as response:
                data = response.read(4 * 1024 * 1024)
            require(hashlib.sha256(data).hexdigest() == checksum, "Upstream download differs: " + name)
            target = stage / "upstream" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        with urllib.request.urlopen(NLTK_URL, timeout=60) as response:
            archive = response.read(8 * 1024 * 1024)
        require(hashlib.sha256(archive).hexdigest() == NLTK_SHA, "NLTK archive differs")
        (stage / "resources").mkdir()
        (stage / "resources/punkt_tab.zip").write_bytes(archive)
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            for name, checksum in NLTK_FILES.items():
                data = source.read(name)
                require(hashlib.sha256(data).hexdigest() == checksum, "NLTK resource differs")
                target = stage / "nltk_data/tokenizers" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        rows = [json.loads(line) for line in (stage / "upstream" / DATA).read_text().splitlines()]
        save(stage / "manifest.json", descriptor(rows))
        checksum = sha(stage / "manifest.json")
        save(stage / "SEALED.json", {"manifest_sha256": checksum})
        load(stage, checksum)
        private(stage)
        stage.rename(out)
    return {"directory": str(out), "manifest_sha256": checksum, "prompts": 541, "instruction_instances": 834,
            "model_calls": 0, "license": "Apache-2.0"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("fetch").add_argument("--out", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--source", type=Path, required=True)
    check.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    result = fetch(args.out) if args.action == "fetch" else load(args.source, args.source_sha)[0]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
