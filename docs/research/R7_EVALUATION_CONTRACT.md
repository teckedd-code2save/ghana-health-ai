# R7 Evaluation Contract

Version: `r7-eval-v1`. This is an offline measurement contract, **not an R7 result**,
a promotion decision, or clinical certification. No evaluation cases or outcomes
are asserted here. Unit-test records are synthetic software fixtures only.

## Scope And Interfaces

`scripts/r7_eval_metrics.py` uses Python 3.11+ and the standard library only. It
does not generate responses, invoke a judge, execute tools, access APIs, train,
or launch accelerators. The main pipeline owns frozen case preparation,
collection, trace authenticity, manifest storage, and reviewer recruitment.

```python
digest(json_value) -> str
parse_json(text) -> Json
validate_rows(rows: list[EvalRow], *, expected_rows_sha256: str) -> None
evaluate_rows(rows: list[EvalRow], *, expected_rows_sha256: str) -> EvaluationResult
judgment_input(row: EvalRow, dimension: str) -> dict
repetition_metrics(output: str, messages: list[Message], *, n=4) -> dict
paired_report(rows, *, expected_rows_sha256, baseline, candidate,
              metric: Metric, seed=17, resamples=2000) -> dict
audit_overlap(training: list[SourceRecord], evaluation: list[SourceRecord], *,
              expected_training_sha256, expected_evaluation_sha256) -> dict
```

The CLI reads a JSON array and prints a report to stdout. It does not write files:

```sh
python -B scripts/r7_eval_metrics.py ROWS.json --expected-sha256 TRUSTED_ROWS_SHA256
python -B -m unittest discover -s scripts -p test_r7_eval_metrics.py -v
```

## Row Contract

The exported `TypedDict` definitions are authoritative. Every field shown is
required except `probe`; unknown fields, duplicate JSON keys, nonfinite values, and duplicate
model/case rows are rejected. No caller-supplied score, `solved`, or `gold` flag is
accepted. These are field/type descriptions, not an executable evaluation record:

```text
EvalRow = {
  version: "r7-eval-v1",
  case: {
    id, scenario_id, source_group_id, domain, input_language, output_language: string,
    mode: "fixed_shared_history" | "independent_rollout", turn: nonnegative integer,
    messages: [{role: "system" | "user" | "assistant" | "tool", content: string}],
    sources: {source_id: frozen source text},
    constraints: [{id: metric name, kind: executable check, expected: JSON}],
    rubrics: {semantic dimension: frozen rubric text},
    probe?: "fresh" | "correction" | "followup" | "topic_change" | "unspecified"
  },
  model_id: checkpoint label, model_sha256: SHA256, protocol_sha256: SHA256,
  messages: [Message],
  response: {
    state: "complete" | "invalid" | "error" | "incomplete" | "blocked" | "pending",
    output: raw response text,
    disposition: "answer" | "abstain" | "refuse" | "unknown",
    executions: [{call: {id, name, arguments: object}, sequence: integer,
                  status: "success" | "error", transaction_id: string | null,
                  confirmation_id: string | null}],
    confirmations: [{id, sequence: integer, accepted: boolean, name, arguments: object}]
  },
  judgments: [{dimension, label: "pass" | "fail" | "abstain" | "unknown",
               judge: {kind: "human" | "model", id, version,
                       input: judgment_input(row, dimension), input_sha256: SHA256}}]
}
```

The result shape is:

```text
EvaluationResult = {version, rows_sha256, row_count, unique_cases, source_groups,
  rows: [{case_id, model_id, state, disposition, objective: {check_id:
    {passed, observed, reason, diagnostics?}}, repetition, structured_output_invalid, judgments}],
  strata: [{labels, expected_rows, unique_cases, source_groups, states, dispositions,
    disposition_denominator, structured_invalid_rows, objective, semantic_expected,
    semantic, repetition}],
  semantic_quality_established: false, clinical_certification: false,
  model_judgments_are_gold: false}
```

`case.id` identifies a scheduled response slot, not an attempt or a model copy.
`scenario_id` identifies a conversation; `source_group_id` identifies the original
source family across turns, translations, paraphrases, and derived cases. Domain
and language labels are authored metadata, never inferred from response text.
`disposition` is an explicit collector annotation or structured action, **not a
semantic detector**. Use `unknown` when it is unavailable; freeze the annotation
method in the protocol. Appropriate abstention/refusal can be a correct outcome.

Keep every planned slot, including failed, unattempted, and blocked slots. A
malformed **model response** is a counted nonpass; a malformed **evidence record**
raises `ValueError` and must be repaired before reporting, not silently excluded.
For upstream R5 states such as paused/indeterminate, preserve the original state
in the collection archive and freeze any mapping to this smaller state vocabulary.

### Identity And Trust

Before inference, freeze case IDs, groups, messages, sources, expected checks,
rubrics, model artifact hashes, and protocol. The protocol must identify dataset
revisions, tokenizer/template/runtime, decoding/seed, raw versus guarded policy,
tool schemas/simulator version, disposition annotation, and judge instructions.
The model hash must resolve to the actual foundation/adapter identity.

The collector must verify each row against that pre-inference inventory, then
seal the complete collected rows, including placeholders and review provenance.
The supplied `expected_rows_sha256` must come from that independently trusted
manifest, not a hash recomputed from possibly modified input at verification time.
`digest` uses sorted, compact, UTF-8 JSON with Unicode preserved and finite values.
The row-list order is part of this seal. Adding judgments creates a new sealed
snapshot; do not overwrite prior evidence.

This scorer verifies the supplied seal and internal consistency. It cannot
authenticate fabricated traces, prove a model ran, detect a maliciously resealed
manifest, or discover omitted planned slots without the upstream frozen inventory.
Never accept model-generated execution/confirmation receipts as trusted traces.

## Executable Checks

Constraint IDs name metrics and should have stable meanings across cases. Each
check is recomputed from raw output/trace, never from saved success flags.

| Kind | `expected` | Exact measurement |
| --- | --- | --- |
| `exact_text` | string | Raw string equality, no normalization |
| `json_equal` | any JSON | Strict parsed JSON equality; boolean differs from number |
| `json_field_equal` | `{field, value}` | Exact named JSON field; free explanation remains independently reviewed |
| `citation_ids` | `{allowed: [id], required: [id]}` | Unique output IDs contain required IDs and no unregistered IDs |
| `tool_selection` | ordered tool-name list | Exact selected tool names and count |
| `tool_arguments` | ordered `{id,name,arguments}` list | Exact call identity and JSON arguments |
| `tool_execution` | `null` | Calls exactly match ordered execution receipts and all succeeded |
| `commerce_confirmation` | mutating tool-name list | Every attempted listed mutation has one prior accepted, unused, action/argument-bound confirmation |
| `transaction_claims` | `null` | Every structured transaction claim matches a successful call/transaction receipt |
| `disposition` | disposition string | Collector action matches expected action |
| `category_membership` | list spec below | Exact requested count; all items in the frozen category vocabulary |
| `list_distinctness` | list spec below | Exact requested count; distinct canonical items |
| `correction_uptake` | list spec below | Prior assistant context exists; corrected target count/category/distinctness and no obsolete items |
| `topic_change` | list spec below | Prior assistant context exists; new target count/category/distinctness and no obsolete items |

Structured output checks read fields from a JSON object in `response.output`:
`citations: [source_id]`, `tool_calls: [{id,name,arguments}]`, and
`transaction_claims: [{call_id,transaction_id}]`. A required missing/malformed
field fails. Empty lists are valid only where their contract permits no action.
Exact JSON arguments are type-sensitive and include amounts/items/recipients as
ordinary structured data, not loosely matched substrings.

Use tool-selection, arguments, execution, confirmation, and transaction-claim
checks together for commerce cases. No execution means the confirmation check
alone is vacuously satisfied. A failed execution is not a completed transaction;
even attempted mutating calls require confirmation. Events have unique ordered
sequence numbers. Confirmation reuse is rejected within the recorded trace.
Multi-turn commerce traces must be audited cumulatively by the collector for
cross-turn replay; the per-response check does not authenticate external state.

Registered citation IDs do not establish entailment, source reliability, or
coverage of every claim. Structured transaction checks do not parse unsupported
claims embedded in free prose. Literal checks do not establish meaning, language
adherence, usefulness, or medical safety.

### Category And Transition Diagnostics

The list spec is `{format, field, count, allowed, forbidden, aliases}`.
`format="json_field"` reads one top-level array of strings named by `field`.
`format="numbered_lines"` requires `field=null` and extracts consecutive numbered
lines starting at 1; other prose is **not** semantically scored. Repeated list
numbering, skipped numbers, and absent lists are unparseable, not successes. Use
separate JSON fields/checks for prompts requesting multiple categories.
`count` is the positive requested size. `allowed`/`forbidden` contain normalized
canonical item IDs. `aliases` maps normalized surface forms to those IDs, so
translations or inflections of the same item cannot inflate distinctness.
Freeze and independently review this vocabulary; an unlisted legitimate answer
needs review, not an unsupported semantic "wrong" label.

Per-check diagnostics retain requested/returned/distinct/duplicate/in-category/
unlisted/obsolete item counts, fractions over returned items, prior-list
availability, and prior out-of-category count. Aggregate requested-item coverage
includes invalid/error/unparseable slots, while returned-item fractions never
hide their separate slot denominators. Coverage caps valid distinct items at the
requested size per row; extra answers still fail the exact-count constraint.

Correction/topic endpoints measure **post-instruction target satisfaction** on
predeclared correction/topic-change probes, not causal understanding or inferred
intent. Report previous out-of-category opportunities separately; do not silently
condition comparisons on only one model's earlier mistakes. Keep human/model
semantic review under the corresponding dimensions for open-ended responses.
Use `probe` for fresh, correction, follow-up, and topic-change breakdowns; omitted
metadata is `unspecified`, never guessed from a prompt.

## Semantic Review

Supported rubric dimensions: `meaning`, `multiturn_usefulness`,
`source_grounding`, `citation_support`, `tool_appropriateness`,
`commerce_integrity`, `language_adherence`, `english_retention`,
`abstention_appropriateness`, `refusal_appropriateness`, `medical_safety`,
`category_membership`, `list_distinctness`, `correction_uptake`, `topic_change`.
Only dimensions with frozen case rubrics have expected review denominators.

Preserve negation, quantities, entities, corrections, and intent in meaning review.
Assess useful next steps and context consistency over complete dialogues. Assess
claim/source support and prose transaction claims separately from identifier
checks. Language review should evaluate requested-language adherence and genuine
understandability, including code-switching where specified; no token heuristic
is a native-language judgment. Maintain explicit English cases and compare their
objective and reviewed outcomes independently of Twi or other language gains.

Medical-safety review requires an appropriately qualified reviewer and a frozen
rubric covering unsafe assertions, uncertainty, escalation, and unsupported
claims. This module supplies **no medical advice** and never issues clinical
certification. Passing public medical MCQs or a model judge is not safety evidence
for deployment by itself.

Each judgment stores the exact case/messages/response payload returned by
`judgment_input`, its hash, judge ID, and version. For a model judge, use the model
identifier and pinned model revision; retain the actual rendered judge request,
system instructions, decoding configuration, and raw response in the collection
archive under the frozen judge protocol. For humans, use the reviewer ID and
review-procedure version. Changing output, rubric, sources, or trace invalidates
the attached judgment input. Repeated judgments by the same identity/dimension
are rejected; adjudication must be explicitly versioned.

Model judgments are reported separately by model/version and **never counted as
gold automatically**. Human labels are also attributed observations, not automatic
certification. Disagreements require independent adjudication outside this module.
Judge `abstain`/`unknown` labels are missing decisions, distinct from a model's
response-level abstention/refusal. Report calibration, language qualifications,
blinding, reviewer independence, and inter-reviewer agreement before using labels
for stronger claims. No composite "solved" score is produced.

## Histories, Counts, And Uncertainty

In `fixed_shared_history`, every checkpoint receives exactly `case.messages`.
This measures a controlled next response; it does not measure the consequences
of a checkpoint's own earlier replies. In `independent_rollout`, `case.messages`
contains the frozen cumulative system/user script only. Actual `row.messages`
interleaves that script with **that checkpoint's own** preceding responses.
Turns start at zero and are contiguous, including pending/blocked placeholders.
Continuing after an unsuccessful turn is rejected; an unattempted blocked/pending
or error turn may have an empty actual history. Dynamic user/tool environments
require a separately frozen protocol and adaptation, not mixed-script pairing.

The report contains `rows` with objective pass/observed/reason values, repetition
diagnostics, and original attributed judgments. `strata` contains per-model/mode
totals plus domain x input-language x output-language x probe breakdowns. These totals
and breakdowns overlap and must not be summed together. Top-level case/source
counts deduplicate checkpoint copies; `row_count` does not.

Every stratum retains all states and all dispositions with their all-row
denominator. Objective `pass_rate_all_expected` uses **every applicable slot**,
including errors/invalid/pending as nonpasses; `observed` and `unavailable` are
separate. `structured_invalid_rows` flags nonempty output that cannot supply the
required JSON object, and can overlap a collector `invalid` state. Missing
structured fields fail their checks even if the object itself parses.

Semantic reports retain expected, submitted, missing, pass/fail/abstain/unknown,
decided-only and all-expected pass fractions per judge/version. Unreviewed cases
are not silently promoted to pass. `semantic_expected` remains visible even when
no judgments exist. An all-expected semantic fraction is coverage-sensitive, not
a claim that missing decisions are verified failures.

Repetition uses NFC/casefold/whitespace normalization. Repeated n-gram positions
are occurrences beyond the first, divided by all n-gram positions (default four
word tokens). Short/empty outputs have no n-gram denominator. Cross-turn flags
compare only assistant replies, including nonconsecutive identical replies;
empty output never counts as repetition. This detects mechanical repetition,
not semantic redundancy or correctness. Partial outputs remain diagnostic and
do not become successful responses.

`Metric = {kind: "objective", name: constraint_id}` or
`{kind: "human" | "model", name: dimension, judge_id, judge_version}` selects a
paired endpoint. Pair by case ID, not file order. Shared cases and protocol must
match; raw/guarded or otherwise different protocols require separate reports.
Report baseline-only/candidate-only slots, matched/scored pairs, missing metrics,
and unavailable response counts. Objective errors stay nonpasses; undecided or
missing semantic labels do not enter the decided paired subset.

Each mode gets its own totals and domain/language strata; there is no pooled
fixed/rollout effect. Report row gains/losses/ties **and** source-group gains/losses/
ties. Average paired deltas within each source group, then give each group equal
weight. The seeded percentile bootstrap resamples whole groups, producing a 95%
interval for that group-macro effect, not the row-weighted effect. Fewer than two
scored groups gives `null` uncertainty, never a fake zero-width estimate. Effects
and intervals are fractions; multiply by 100 for percentage points.

Intervals are descriptive for the supplied sample, unadjusted for multiple
comparisons; no significance/promotion label is emitted. Small group counts,
source-order selection, missing pairs/judgments, and overlap limit generalisation.
Do not interpret an English regression as absent merely because an interval
includes zero. Predeclare endpoints and retention margins before execution.

## Train/Eval Separation

`SourceRecord = {id: string, source_group_id: string, text: string}`. Supply
independently hash-bound complete inventories. `audit_overlap` reports evaluation
IDs intersecting training by source ID, source group, exact text, and normalized
text. Counts refer to supplied source records, not unique people or unseen cases.
Empty inventories never establish disjointness.

Include all consumed R5/R6/R7 training ancestry, repairs, targets, translations,
derived views, feedback, and validation/development exposure in the audit scope.
Protect whole source families before splitting. Store inventory coverage,
upstream revisions and hashes, all filtering decisions, near-duplicate/semantic
review evidence, and known exclusions in the main pipeline's frozen manifest.
Do not repair-train on evaluation disagreements and then retain the same cases
as an independent holdout. Separate development, public benchmark, fixed-history,
independent-dialogue, and genuinely source-separated fresh evaluation scopes.

Disjoint supplied inventories do not prove comprehensive separation. The result
explicitly leaves inventory coverage unverified, near-duplicate review unassessed,
foundation pretraining overlap unknown, and independent-holdout status false.
Known public benchmark exposure must be disclosed, including the R6 status
document's PubMedQA partition/overlap limitation.

## Saved R6 Failure Taxonomy

Read-only evidence inspected in the shared worktree on 17 November 2026:
`tmp/understanding-r6-workbench/529200bb9482784bf7045985de3ffda7a712055978ec9f302f19dfeec4ed670a/workbench.sqlite3`.
Session `18d13da7638848258cd345c3b5f163d2` has four saved turns, with wrong-category
lists in the category request, explicit correction, and fish follow-up. Those
lists reuse dogs/cattle/sheep for requested birds/fish. Its final saved input is
568 content characters. Session `d9bd93ed80fe4fb0a50f3dbbaa418aab` starts fresh
with the same corrective bird prompt and returns `Bird` three times; its later
English follow-up supplies three distinct names. These are inspected observations,
not representative benchmark rates or an R7 result.

The saved policy is independent model histories; the main investigation reports
that the independent-history audit passed. Both sessions have the guard enabled
with no trigger on these replies. `response_loop_guard.py` begins its passage
check at 32 words; short list degeneracy can fall below that threshold. A lack of
guard trigger, EOS completion, or low four-gram repetition therefore does not
establish answer quality. Nothing here establishes that leaked history, a long
context, or the guard caused the underlying model error.

Use four separate failure classes: category substitution, short-list duplicate
collapse, correction non-uptake, and obsolete-category persistence after topic
change. Root-cause data repair must investigate source/target supervision and
language-to-category/task binding rather than treating these observations as
proof of one cause. Protect the reported sessions as development evidence; do not
label derived repairs or paraphrases a fresh independent holdout.

The main pipeline should compare R7 with R6 on newly source-separated fresh
requests, controlled corrections, independent follow-ups, topic switches,
paraphrases, and English-retention controls. Freeze raw/guarded protocols and
report each probe/language separately. All model/judge execution remains on
Modal; this sidecar's local runs are CPU-only deterministic tests and saved-record
inspection. No new model evaluation has been executed by this sidecar.

## Frozen Behavior Inputs

`scripts/build_r7_behavior_eval.py` builds the private input pack at
`tmp/r7-20261117/behavior-eval-v1/`. Verified manifest SHA256:
`8029612cea3110d31d9b8a1f717e0f43a38148c613c739341ae46f50d3a68e13`.
This pack contains inputs and expected constraints, **not model predictions**.

| Artifact | Contract |
| --- | --- |
| `cases.jsonl` | 64 `EvalCase` rows, all `fixed_shared_history` |
| `scenarios.jsonl` | 16 `{id,source_group_id,language,system_message,user_turns,expected_per_turn}` objects; `expected_per_turn` contains three full independent-rollout `EvalCase` rows |
| `reused-regressions.jsonl` | Five fixed `EvalCase` rows from the two saved sessions; never merged into fresh counts |
| `source-scenarios.json` | Eight authored semantic roots, bilingual facts/queries and explicit answers |
| `held-content.jsonl` | Deduplicated protection records with group IDs, case memberships, text and `training_eligible=false` |
| `r6-overlap.json` | Hash-verified R6 artifact inventory and declared lexical/group overlap checks |
| `regression-provenance.json` | Saved session/round/input/output hashes; no new model results |
| `manifest.json`, `FROZEN.json` | Artifact/code hashes, scope, computed counts and manifest seal |

Eight families cover category membership, reference resolution, entity binding,
time ordering, quantity updates, negation/exclusion, spatial relations, and state
revision. They produce **112 derived fresh test views, not 112 independent
scenarios**: 64 fixed views plus 48 independent-rollout turns, with 56 views per
language. Fresh/correction/follow-up/topic-change view counts are 32/16/32/32;
the separate reused scope has five views in two groups. All language and history
variants of a source scenario share one protected group. There is no training
partition or prompt-only train/test split in this pack.

The R6 scan checked 256 fresh protected text records against 6,290 registered
collection records and 8,373 prepared-view witnesses, with no matches under the
declared normalized-exact, 8-word-shingle Jaccard >=0.85, and group-identity checks.
These are inventory counts, not unique-source or actually-consumed-view counts.
The machine audit's witness-scope label does not establish actual training
consumption. The checked R6 manifest is
`f8cdda46dc7418060bdad254add35048d6eebdb1b49aa3c5aad47050ba2830d0`.
This supports the limited scanned-pack freshness statement only; semantic
overlap and foundation pretraining exposure remain uncertified.

For a fixed request, `remote_fixed(case)` emits only `{id,messages}`. For an
independent rollout, `remote_rollout(scenario)` emits only
`{id,system_message,user_turns}`. Never send `expected_per_turn`, constraints,
rubrics, or source-scenario answers to the worker. Supplied source facts in the
user prompt are intentional inputs, not hidden expected outputs. Fixed
correction histories intentionally contain wrong answers; independent rollouts
must instead append the selected checkpoint's own outputs. Preserve actual
model-visible messages in every response record and emit all failed/pending
slots. Execution, unchanged frozen model-bundle verification, and Modal GPU
scheduling belong to the main pipeline.

Builds refuse existing output directories. The verifier checks the independent
manifest hash, artifact/code hashes, schema, generated source-to-view consistency,
view/group counts and complete held-content coverage. Synthetic tests check
byte-identical rebuilds, duplicate semantic sources under renamed IDs, matched
languages, contaminated histories, stale R6 evidence, tampering and gold-free
worker projections. No native-language, clinical, or untouched-final-test status
is granted by these checks.

## Relationship To R5/R6

This design follows R5's `run_understanding_r5_evaluation.py` denominator and
review separation, `build_understanding_evaluation.py` strict objective/review
binding, and `audit_understanding_r2_feedback.py` cross-turn repetition checks.
It does not alter them. `R6-STATUS.md` withdraws the old no-English-regression
claim and explicitly does not establish improved meaning, repetition, safety, or
dialogue usefulness. This sidecar adds measurement capacity, not evidence that
those issues are solved. No historical benchmark results are rewritten.
