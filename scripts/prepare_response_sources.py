"""Download pinned public text sources for the response-training experiment."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "tmp/response-adaptation-v2"
SOURCES = {
    "ghana_chat": {
        "repo": "michsethowusu/ghana-chat-corpus-ak",
        "revision": "56cb228a73f2b1e9226d20bca8f2fa9f6fc9b94b",
        "license": "cc-by-nc-4.0", "attribution": "Mich-Seth Owusu and Ghana NLP Community",
        "files": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
        "limitations": "Synthetic English conversations with machine-translated Twi; joined assistant replies are not one answer. Not human gold.",
    },
    "everyday": {
        "repo": "HuggingFaceTB/everyday-conversations-llama3.1-2k",
        "revision": "14f543216b9ba42b6b951dc5bd199460d193b162",
        "license": "apache-2.0", "attribution": "Hugging Face",
        "files": ["data/train_sft-00000-of-00001.parquet", "data/test_sft-00000-of-00001.parquet"],
        "limitations": "Llama-3.1-70B synthetic English conversations; repeated greeting turn must not dominate supervision.",
    },
    "apigen": {
        "repo": "argilla/apigen-function-calling",
        "revision": "170a8a7e0832101f2495a00c9ad9ba1821d3a9b0",
        "license": "cc-by-4.0", "attribution": "Argilla, Salesforce xLAM and Synth-APIGen contributors",
        "files": ["data/train-00000-of-00001.parquet"],
        "limitations": "Synthetic function selection/arguments, not executed result-to-answer trajectories. Validate schema conversions; never execute arbitrary dataset functions.",
    },
}


def main():
    manifest = {"created": datetime.now(timezone.utc).isoformat(), "sources": {},
                "private_research_only": True, "human_validated": False}
    for key, source in SOURCES.items():
        folder = FOLDER / "sources" / key
        files = []
        for filename in ["README.md", *source["files"]]:
            path = Path(hf_hub_download(source["repo"], filename, repo_type="dataset",
                        revision=source["revision"], local_dir=folder))
            with path.open("rb") as handle:
                sha = hashlib.file_digest(handle, "sha256").hexdigest()
            files.append({"file": filename, "bytes": path.stat().st_size, "sha256": sha})
            print(json.dumps({"source": key, **files[-1]}), flush=True)
        manifest["sources"][key] = {**source, "files": files}
        (FOLDER / "sources.partial.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (FOLDER / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
