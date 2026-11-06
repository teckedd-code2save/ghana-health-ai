# R2/R3 Research Reproducibility

This public checkout preserves code and synthetic unit fixtures, not training
data, live conversations, credentials, or model artifacts. It is a progress
backup for issue #43, not a model release or production promotion.

Status reported on 6 November 2026: R3 training completed 105 optimizer updates
in approximately 1,179 seconds; evaluation is running. Completion is not evidence
of language quality, clinical safety, or improvement over another checkpoint.

## Source Inputs Required

- R2: the exact prepared source snapshot, R1 control pack, frozen evaluation,
  tokenizer assets, and pinned base-model revision expected by the builders.
- R3: the verified R2 pack, versioned conversation datasets, manifest-bound
  admission decisions, run selection, and frozen development evaluation.
- Inference/evaluation: completed training receipts, terminal state, consumption
  evidence, full adapter readback, and the checksum-bound R2/R3 inference bundles.

Obtain these separately through the authorized private artifact store. Dataset
IDs, source IDs, source/record hashes, original and translation provenance,
review/admission state, and request/cost references must survive restoration.
Unresolved candidates are not admitted training inputs. Agent review is not
native-human certification. Never substitute fresh data for a pinned input or
change checksums to make a mismatched restoration pass.

The 12 helpers in `run_understanding_r2.FROZEN` retain their original bytes.
Four additional imported helpers are included for clean-checkout imports.
Set the source-root override below rather than relying on legacy host defaults.
The host-bound paid repair pipeline and private regression fixtures are not part
of this public backup; no paid repair service is needed for the unit tests.

## Offline Code Checks

The CPU test environment used for this backup is Python 3.13.9, PyTorch 2.12.0,
NumPy 2.3.5, and Gradio 6.17.3. Dependency installation needs a package source;
after installation, tests use synthetic temporary inputs and mocked services.
No model download, Modal submission, training, or paid API call is required.

```sh
python3.13 -m venv tmp/research-venv
. tmp/research-venv/bin/activate
python -m pip install torch==2.12.0 numpy==2.3.5 gradio==6.17.3
export GHA_R2_SOURCE_ROOT="$PWD"
export PYTHONPATH="$PWD/scripts:$PWD/modal/train"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1
python -B -m unittest \
  test_prepare_understanding_r2 test_run_understanding_r2 \
  test_prepare_understanding_r2_evaluation test_checkpoint_understanding_r2 \
  test_r2_inference test_r2_comparison test_understanding_r2_demo \
  test_understanding_r2_workbench test_audit_understanding_r2_feedback \
  test_admit_understanding_r3_repairs test_prepare_understanding_r3 \
  test_run_understanding_r3 test_checkpoint_understanding_r3 \
  test_prepare_understanding_r3_evaluation test_run_understanding_r3_evaluation \
  test_prepare_understanding_r3_inference test_understanding_r3_demo
```

Production research workers use Python 3.11 and their own pinned GPU runtime in
`run_understanding_r2.RUNTIME` and the sealed inference manifest. The CPU test
environment is not a replacement for that runtime. Remote operations also need
the Modal SDK and separately provisioned credentials; never put credentials in
this checkout. Existing application CI does not run these Python tests.

## Restored Artifact Verification

With private paths and expected digests supplied in environment variables:

```sh
python -B scripts/run_understanding_r3.py verify --pack "$R3_PACK"
python -B scripts/run_understanding_r3_evaluation.py verify \
  --execution "$R3_EVALUATION_EXECUTION" \
  --execution-sha "$R3_EVALUATION_EXECUTION_SHA"
python -B scripts/prepare_understanding_r3_inference.py --help
```

These commands do not submit remote work. Launches, evaluation jobs, uploads,
and deployments require separate authorization and verified immutable inputs.
The gateway and app integration remain outside this backup's scope.

## Publication Boundary

Keep research datasets, request/response caches, packs, weights, owner regression
text, and internal run notes out of public Git. Authorized private Modal
evaluation inputs may include held-out owner regressions; that backup is not
owner-free. Raw SQLite stores, freeform reviews, mail, and credentials are
excluded from general artifact backups. Never copy or reconstruct the canonical
repair spend ledger. A storage readback receipt proves byte integrity, not
publication permission, model quality, or an independent backup of all tensors.
