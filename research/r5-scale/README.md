# R5 matched evaluation implementation

These are the exact workers and offline scorer used for the 11 November 2026
[checkpoint comparison](../../docs/TWI_UNDERSTANDING_SCALE_EVALUATION.md).
The [aggregate JSON](../../docs/twi-scale-evaluation-20261111.json) binds the
worker, case, inference bundle and raw-result hashes for each measurement.
The public repository does not contain the private inputs or outputs.

- `reference_choice_worker.py` scores A and B on the same prompt, then swaps
  reference order. It neither produces nor judges a free-form translation.
- `context_rollout_worker.py` generates each checkpoint's replies and feeds
  back only that checkpoint's own history. It binds already completed turns
  from the interrupted run by hash; invalid history is not cleaned or replaced.
- `report_results.py` verifies the frozen inputs and worker identity, scores
  the saved outputs and renders the private evidence viewer. It requires
  Python and NumPy, not model access or a network call.

**Reproduce the published numbers without GPU charges:** restore the private
11 November handoff and run `python3 report_results.py` from its restored root.
Compare `results-v1/metrics.json` to the published aggregate JSON. Do not commit
the resultant HTML: it embeds the private examples and replies. A bare Git
checkout alone cannot recreate these measurements.

Re-executing inference is separate: the workers are archived byte-for-byte,
expect sibling case folders in the handoff layout and a verified R5 inference
bundle in the adjacent R5 run directory. They require project-authorised Modal
compute and private model volumes. Do not run a worker to regenerate a report.
