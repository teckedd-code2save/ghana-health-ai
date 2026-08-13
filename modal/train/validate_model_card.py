"""
Validate Ghana Health AI Hugging Face model-card metadata locally.

This catches the exact failure that broke earlier pushes: YAML metadata fields
must use Hub-valid dataset ids, while config/split details belong in prose.
"""

from __future__ import annotations

import re
import sys

from model_card import build_model_card_md


DATASET_ID_RE = re.compile(r"^(?:[\w-]+/)?[\w.-]+$")


def _front_matter(md: str) -> str:
    parts = md.split("---", 2)
    if len(parts) < 3:
        raise AssertionError("missing YAML front matter")
    return parts[1]


def _assert_no_invalid_dataset_ids(front: str) -> None:
    dataset_lines: list[str] = []
    capture = False
    for raw in front.splitlines():
        line = raw.rstrip()
        if line == "datasets:":
            capture = True
            continue
        if capture and line.startswith("- "):
            dataset_lines.append(line[2:].strip())
            continue
        if capture and line and not line.startswith(" "):
            capture = False

    for dataset_id in dataset_lines:
        if not DATASET_ID_RE.fullmatch(dataset_id):
            raise AssertionError(f"invalid dataset id in datasets metadata: {dataset_id}")

    for match in re.finditer(r"\n      type: (.+)", front):
        dataset_id = match.group(1).strip()
        if not DATASET_ID_RE.fullmatch(dataset_id):
            raise AssertionError(f"invalid model-index dataset type: {dataset_id}")


def main() -> int:
    md = build_model_card_md(
        repo_id="teckedd/gha-card-validation",
        task="automatic-speech-recognition",
        language=["tw", "ak"],
        base_model="openai/whisper-small",
        metrics={"wer": 0.3044, "cer": 0.1062},
        datasets=[
            "google/WaxalNLP:aka_asr",
            "fsicoli/common_voice_22_0:tw",
            "fsicoli/common_voice_22_0:en",
            "ghananlpcommunity/twi-speech-text-multispeaker-16k:default",
        ],
        summary="Validation card for Ghana Health AI model-card metadata.",
    )
    _assert_no_invalid_dataset_ids(_front_matter(md))
    print("model-card metadata ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
