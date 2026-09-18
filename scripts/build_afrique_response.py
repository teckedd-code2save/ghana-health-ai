"""Source-preserving instruction SFT mix without the rejected general Twi translations."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/afrique-response-v3/corpus"


def main():
    spec = importlib.util.spec_from_file_location("response_core", ROOT / "modal/train/response_adaptation_core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    parent = ROOT / "tmp/response-adaptation-v2/corpus"
    original, splits = core.load_inputs(parent)
    OUT.mkdir(parents=True, exist_ok=False)
    counts, artifacts, exclusions = Counter(), [], []
    for split, rows in splits.items():
        retained = []
        for row in rows:
            if row["task"] == "general_source_response":
                exclusions.append({"id": row["id"], "source": row["source"], "source_id": row["source_id"],
                                   "reason": "Uncalibrated multi-hop general-answer translations excluded as a source family"})
                continue
            retained.append(row)
            counts[split + ":" + row["task"] + ":" + row["language"]] += 1
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained).encode()
        filename = split + ".jsonl"
        (OUT / filename).write_bytes(data)
        artifacts.append({"file": filename, "rows": len(retained), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {**original, "experiment": "afrique_response_sft_v3",
                "base_model": "McGill-NLP/AfriqueQwen3.5-4B-50Langs", "base_revision": "ea443ca5e6674e17c271fb66e54e3282fe78d21a",
                "tokenizer_model": "Qwen/Qwen3.5-4B", "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "counts": dict(counts), "artifacts": artifacts, "unique_sources": sum(a["rows"] for a in artifacts),
                "parent_manifest_sha256": hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest(),
                "excluded_rows": len(exclusions), "limitations": [
                    "Research instruction adaptation of a Twi-including CPT base, not a clinically verified assistant.",
                    "All general_source_response multi-hop translations are excluded; original source records unchanged.",
                    "General conversation supervision is predominantly English; Twi conversation breadth remains a gap.",
                    "AfriHealth source answers are unchanged; clinical correctness is not independently certified.",
                    "Tool targets cover calls, not executed trajectories; no web or commerce actions are connected in evaluation.",
                    "Original source terms and attribution remain, including noncommercial/share-alike restrictions.",
                    "This changes foundation and source mixture; it is not a single-variable causal ablation.",
                ]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "exclusions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in exclusions))
    print(json.dumps({"folder": str(OUT), "counts": counts, "excluded": len(exclusions)}, indent=2))


if __name__ == "__main__":
    main()
