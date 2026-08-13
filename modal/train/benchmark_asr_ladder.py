"""
Launch the Ghana Health AI ASR benchmark ladder on Modal.

Default mode prints the exact commands. Use --execute to run them.
Use --include-train only after evals show which branch deserves more credits.

Examples:
  python modal/train/benchmark_asr_ladder.py
  python modal/train/benchmark_asr_ladder.py --execute --max-samples 500
  python modal/train/benchmark_asr_ladder.py --execute --include-train --no-wait
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]


def eval_jobs(max_samples: int, no_wait: bool, include_english: bool) -> list[Job]:
    wait_flag = ["--no-wait"] if no_wait else []
    run_prefix = ["modal", "run", "--detach"] if no_wait else ["modal", "run"]
    jobs = [
        Job(
            "whisper-v6-greedy",
            [
                *run_prefix,
                "modal/train/eval_asr.py",
                "--model-id",
                "teckedd/gha-whisper-small-twi-v6",
                "--max-samples",
                str(max_samples),
                "--num-beams",
                "1",
                "--streaming",
                *wait_flag,
            ],
        ),
        Job(
            "whisper-v6-beam5",
            [
                *run_prefix,
                "modal/train/eval_asr.py",
                "--model-id",
                "teckedd/gha-whisper-small-twi-v6",
                "--max-samples",
                str(max_samples),
                "--num-beams",
                "5",
                "--streaming",
                *wait_flag,
            ],
        ),
        Job(
            "whisper-small-base",
            [
                *run_prefix,
                "modal/train/eval_asr.py",
                "--model-id",
                "openai/whisper-small",
                "--max-samples",
                str(max_samples),
                "--num-beams",
                "1",
                "--streaming",
                *wait_flag,
            ],
        ),
        Job(
            "dondo-w2v-bert",
            [
                *run_prefix,
                "modal/train/eval_dondo_asr.py",
                "--model-id",
                "KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en",
                "--max-samples",
                str(max_samples),
                *wait_flag,
            ],
        ),
    ]
    if include_english:
        english_common = [
            "--dataset-name",
            "fsicoli/common_voice_22_0",
            "--dataset-config",
            "en",
            "--split",
            "test",
            "--language",
            "en",
            "--streaming",
            "--trust-remote-code",
        ]
        jobs.extend(
            [
                Job(
                    "english-retention-v6-beam5",
                    [
                        *run_prefix,
                        "modal/train/eval_asr.py",
                        "--model-id",
                        "teckedd/gha-whisper-small-twi-v6",
                        *english_common,
                        "--max-samples",
                        str(max_samples),
                        "--num-beams",
                        "5",
                        *wait_flag,
                    ],
                ),
                Job(
                    "english-baseline-whisper-small-beam5",
                    [
                        *run_prefix,
                        "modal/train/eval_asr.py",
                        "--model-id",
                        "openai/whisper-small",
                        *english_common,
                        "--max-samples",
                        str(max_samples),
                        "--num-beams",
                        "5",
                        *wait_flag,
                    ],
                ),
            ]
        )
    return jobs


def train_jobs(no_wait: bool) -> list[Job]:
    wait_flag = ["--no-wait"] if no_wait else []
    run_prefix = ["modal", "run", "--detach"] if no_wait else ["modal", "run"]
    return [
        Job(
            "whisper-small-waxal-proof-streamed",
            [
                *run_prefix,
                "modal/train/train_asr.py",
                "--base-model",
                "openai/whisper-small",
                "--run-name",
                "v7-small-waxal-proof-streamed",
                "--max-steps",
                "80",
                "--train-limit",
                "256",
                "--eval-limit",
                "64",
                "--no-use-extra-data",
                "--no-full-test-after",
                "--push-repo",
                "",
                *wait_flag,
            ],
        ),
        Job(
            "whisper-small-balanced-extra-proof-streamed",
            [
                *run_prefix,
                "modal/train/train_asr.py",
                "--base-model",
                "openai/whisper-small",
                "--run-name",
                "v7-small-balanced-extra-proof-streamed",
                "--max-steps",
                "80",
                "--waxal-weight",
                "0.50",
                "--train-limit",
                "512",
                "--eval-limit",
                "64",
                "--no-full-test-after",
                "--push-repo",
                "",
                *wait_flag,
            ],
        ),
        Job(
            "whisper-small-balanced-v7-lite",
            [
                *run_prefix,
                "modal/train/train_asr.py",
                "--base-model",
                "openai/whisper-small",
                "--run-name",
                "v7-small-balanced-lite-no-en-regression",
                "--max-steps",
                "500",
                "--waxal-weight",
                "0.50",
                "--train-limit",
                "1200",
                "--eval-limit",
                "150",
                "--no-full-test-after",
                "--push-repo",
                "teckedd/gha-whisper-small-twi-en-balanced-v7-lite",
                *wait_flag,
            ],
        ),
        Job(
            "whisper-small-balanced-v7-lite-frozen",
            [
                *run_prefix,
                "modal/train/train_asr.py",
                "--base-model",
                "openai/whisper-small",
                "--run-name",
                "v7-small-balanced-lite-frozen-no-en-regression",
                "--max-steps",
                "500",
                "--learning-rate",
                "8e-6",
                "--freeze-encoder",
                "--waxal-weight",
                "0.50",
                "--train-limit",
                "1200",
                "--eval-limit",
                "150",
                "--no-full-test-after",
                "--push-repo",
                "teckedd/gha-whisper-small-twi-en-balanced-v7-lite-frozen",
                *wait_flag,
            ],
        ),
        Job(
            "whisper-small-balanced-v7",
            [
                *run_prefix,
                "modal/train/train_asr.py",
                "--base-model",
                "openai/whisper-small",
                "--run-name",
                "v7-small-balanced-no-en-regression",
                "--max-steps",
                "3000",
                "--waxal-weight",
                "0.45",
                "--push-repo",
                "teckedd/gha-whisper-small-twi-en-balanced-v7",
                *wait_flag,
            ],
        ),
        Job(
            "whisper-small-balanced-v7-frozen-encoder",
            [
                *run_prefix,
                "modal/train/train_asr.py",
                "--base-model",
                "openai/whisper-small",
                "--run-name",
                "v7-small-balanced-frozen-no-en-regression",
                "--max-steps",
                "2500",
                "--learning-rate",
                "8e-6",
                "--freeze-encoder",
                "--waxal-weight",
                "0.45",
                "--push-repo",
                "teckedd/gha-whisper-small-twi-en-balanced-v7-frozen",
                *wait_flag,
            ],
        ),
        Job(
            "whisper-medium-balanced-v7",
            [
                *run_prefix,
                "modal/train/train_asr.py",
                "--base-model",
                "openai/whisper-medium",
                "--run-name",
                "v7-medium-balanced-no-en-regression",
                "--max-steps",
                "3000",
                "--batch-size",
                "4",
                "--grad-accum",
                "8",
                "--waxal-weight",
                "0.45",
                "--push-repo",
                "teckedd/gha-whisper-medium-twi-en-balanced-v7",
                *wait_flag,
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--skip-english", action="store_true")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--no-wait", action="store_true")
    argv = [arg for arg in sys.argv[1:] if arg != "--"]
    args = parser.parse_args(argv)

    jobs = eval_jobs(args.max_samples, args.no_wait, include_english=not args.skip_english)
    if args.include_train:
        jobs.extend(train_jobs(args.no_wait))

    for job in jobs:
        print(f"\n[{job.name}]")
        print(" ".join(job.command))
        if args.execute:
            subprocess.run(job.command, check=True)


if __name__ == "__main__":
    main()
