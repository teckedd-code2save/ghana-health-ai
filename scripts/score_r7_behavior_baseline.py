#!/usr/bin/env python3
"""Offline conversion of a sealed raw baseline snapshot into attributed CPU metrics."""
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

import build_r7_behavior_eval as inputs
import r7_eval_metrics as metrics
import run_r7_behavior_baseline as launcher

VERSION = 'r7-behavior-score-v1'
REVIEW_VERSION = 'r7-behavior-agent-review-v1'
SCOPES = ('fresh', 'reused')
LABELS = ('pass', 'fail', 'abstain', 'unknown')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def coverage(rows):
    states = Counter(row['response']['state'] for row in rows)
    return {'expected_rows': len(rows), 'finalized_rows': len(rows) - states['pending'],
            'states': {state: states[state] for state in metrics.STATES},
            'source_groups': len({row['case']['source_group_id'] for row in rows}),
            'by_mode': {mode: sum(row['case']['mode'] == mode for row in rows)
                        for mode in ('fixed_shared_history', 'independent_rollout')},
            'rows_sha256': metrics.digest(rows)}


def answer_value(output):
    try:
        value = metrics.parse_json(output)
    except ValueError:
        return None
    return value.get('answer') if type(value) is dict else None


def controlled_diagnostics(row, rows_by_turn):
    """Exact symbols only; neither prose interpretation nor a semantic pass."""
    case = row['case']
    expected = next((c['expected']['value'] for c in case['constraints']
                     if c['id'] == 'answer' and c['kind'] == 'json_field_equal'), None)
    actual = answer_value(row['response']['output'])
    prior = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
    previous = answer_value(prior[-1]) if prior else None
    available = expected is not None and row['response']['state'] == 'complete'
    exact = bool(available and metrics.digest(actual) == metrics.digest(expected)) if expected is not None else None
    correction = case.get('probe') == 'correction' and expected is not None
    changed = case.get('probe') in ('followup', 'topic_change') and expected is not None
    prior_errors = []
    if case['mode'] == 'independent_rollout':
        for turn in range(case['turn']):
            earlier = rows_by_turn.get((case['scenario_id'], turn))
            if earlier is None:
                continue
            target = next((c['expected']['value'] for c in earlier['case']['constraints']
                           if c['id'] == 'answer' and c['kind'] == 'json_field_equal'), None)
            if target is not None and metrics.digest(answer_value(earlier['response']['output'])) != metrics.digest(target):
                prior_errors.append(earlier['case']['id'])
    return {'expected_answer': expected, 'actual_answer': actual, 'strict_answer_match': exact,
            'scalar_singleton_equivalent_not_a_pass': bool(available and type(expected) is list and len(expected) == 1
                and type(actual) is not list and metrics.digest(actual) == metrics.digest(expected[0])),
            'previous_answer': previous, 'injected_wrong_prior': bool(correction and previous is not None and metrics.digest(previous) != metrics.digest(expected)),
            'copied_injected_wrong_answer': bool(correction and previous is not None and metrics.digest(previous) != metrics.digest(expected)
                and metrics.digest(actual) == metrics.digest(previous)),
            'correction_uptake_exact': exact if correction else None,
            'constraint_change_exact': exact if changed else None,
            'earlier_rollout_answer_failures': prior_errors,
            'semantic_pass_inferred': False}


def bind_review(rows, row_sources, progress, decisions, execution_sha):
    """Bind supplied agent decisions to the exact reviewed snapshot; never invent labels."""
    if 'profiles' in decisions:
        # Profiles compress explicitly enumerated decisions, never select rows by score/language.
        metrics.fields(decisions, 'version execution_sha256 progress_snapshot_sha256 reviewer policy profiles records')
        cases = {row['case']['id']: row['case'] for values in rows.values() for row in values}
        expanded = []
        for reference in decisions['records']:
            metrics.fields(reference, 'case_id profile')
            metrics.require(reference['case_id'] in cases and reference['profile'] in decisions['profiles'],
                            'Unknown review profile/case')
            profile = decisions['profiles'][reference['profile']]
            metrics.fields(profile, 'whole_answer_label full_answer_reviewed native_review_required negation_refusal reason default_label labels')
            metrics.require(profile['default_label'] in LABELS and type(profile['labels']) is dict
                            and set(profile['labels']) <= set(metrics.DIMENSIONS)
                            and all(label in LABELS for label in profile['labels'].values()), 'Invalid review label profile')
            labels = {dimension: profile['labels'].get(dimension, profile['default_label'])
                      for dimension in cases[reference['case_id']]['rubrics']}
            expanded.append({**{key: value for key, value in profile.items() if key not in ('default_label', 'labels')},
                             'case_id': reference['case_id'], 'labels': labels})
        decisions = {**{key: value for key, value in decisions.items() if key != 'profiles'}, 'records': expanded}
    metrics.fields(decisions, 'version execution_sha256 progress_snapshot_sha256 reviewer policy records')
    metrics.require(decisions['version'] == REVIEW_VERSION and decisions['execution_sha256'] == execution_sha
                    and decisions['progress_snapshot_sha256'] == progress['snapshot_sha256'], 'Review snapshot identity differs')
    reviewer = decisions['reviewer']
    metrics.fields(reviewer, 'kind id version version_disclosed native_twi_reviewer')
    metrics.require(reviewer['kind'] == 'model' and metrics.nonempty(reviewer['id']) and metrics.nonempty(reviewer['version'])
                    and type(reviewer['version_disclosed']) is bool and reviewer['native_twi_reviewer'] is False,
                    'Attributed agent model/version required; no native certification')
    metrics.require(reviewer['version_disclosed'] or reviewer['version'] == 'not_exposed_by_runtime',
                    'Undisclosed judge version must be explicit')
    metrics.require(metrics.nonempty(decisions['policy']) and type(decisions['records']) is list, 'Review policy/records required')
    flat = [row for scope in SCOPES for row in rows[scope]]
    by_id = {row['case']['id']: row for row in flat}
    sources = {source['case_id']: source for source in row_sources}
    raw_indices = {record['id']: index for index, record in enumerate(progress['results'])}
    by_turn = {(row['case']['scenario_id'], row['case']['turn']): row for row in flat
               if row['case']['mode'] == 'independent_rollout'}
    records, seen = [], set()
    for decision in decisions['records']:
        metrics.fields(decision, 'case_id whole_answer_label full_answer_reviewed native_review_required negation_refusal reason labels')
        case_id = decision['case_id']
        metrics.require(case_id in by_id and case_id not in seen, 'Duplicate or unknown review case')
        seen.add(case_id)
        row, source = by_id[case_id], sources[case_id]
        raw_index = raw_indices.get(source['remote_slot_id'])
        if raw_index is not None:
            raw_record = progress['results'][raw_index]
            metrics.require(row['response']['output'] == raw_record['output'] and row['messages'] == raw_record['messages']
                            and row['response']['state'] == raw_record['state'], 'Review row differs from referenced raw output')
        else:
            metrics.require(row['response']['state'] == 'pending' and row['response']['output'] == '',
                            'Review references an absent raw record')
        metrics.require(decision['whole_answer_label'] in LABELS and metrics.nonempty(decision['reason'])
                        and type(decision['full_answer_reviewed']) is bool and type(decision['native_review_required']) is bool
                        and decision['negation_refusal'] in ('clear', 'present', 'unknown'), 'Invalid full-answer review')
        metrics.require(type(decision['labels']) is dict and set(decision['labels']) == set(row['case']['rubrics'])
                        and all(label in LABELS for label in decision['labels'].values()), 'Review rubric coverage differs')
        if decision['whole_answer_label'] == 'pass':
            metrics.require(decision['full_answer_reviewed'] and decision['negation_refusal'] == 'clear'
                            and not decision['native_review_required'] and row['response']['state'] == 'complete',
                            'Whole-answer pass requires full review and cleared negation/refusal')
        if row['case']['output_language'] == 'tw':
            metrics.require(decision['native_review_required'] and decision['labels'].get('language_adherence') == 'unknown',
                            'Twi native language review is unavailable, not an implicit pass')
        if not decision['full_answer_reviewed'] or row['response']['state'] != 'complete':
            metrics.require(decision['whole_answer_label'] == 'unknown'
                            and all(label == 'unknown' for label in decision['labels'].values()),
                            'Unreviewed/unavailable answers must remain unknown')
        row['judgments'] = []
        for dimension, label in sorted(decision['labels'].items()):
            payload = metrics.judgment_input(row, dimension)
            row['judgments'].append({'dimension': dimension, 'label': label, 'judge': {
                'kind': reviewer['kind'], 'id': reviewer['id'], 'version': reviewer['version'],
                'input': deepcopy(payload), 'input_sha256': metrics.digest(payload)}})
        records.append({**deepcopy(decision), 'scope': source['scope'],
                        'raw_output_sha256': hashlib.sha256(row['response']['output'].encode('utf-8')).hexdigest(),
                        'raw_ref': {'artifact': 'raw-progress.json', 'snapshot_sha256': progress['snapshot_sha256'],
                                    'slot_id': source['remote_slot_id'], 'record_index': raw_indices.get(source['remote_slot_id'])},
                        'controlled': controlled_diagnostics(row, by_turn), 'judgments': deepcopy(row['judgments'])})
    metrics.require(seen == set(by_id), 'Review must account for every planned fresh and reused row')
    records.sort(key=lambda record: record['case_id'])
    return {'version': REVIEW_VERSION, 'reviewer': reviewer, 'policy': decisions['policy'], 'records': records,
            'model_judgments_are_gold': False, 'clinical_certification': False}


def review_summary(rows, reports, review):
    records = {record['case_id']: record for record in review['records']}
    grouped = {}
    for scope in SCOPES:
        scores = {score['case_id']: score for score in reports[scope]['rows']}
        for row in rows[scope]:
            case = row['case']
            labels = {'scope': scope, 'language': case['input_language'], 'family': case['domain'],
                      'mode': case['mode'], 'probe': case.get('probe', 'unspecified')}
            for axes in (('scope',), ('scope', 'language'), ('scope', 'family'), ('scope', 'mode'),
                         ('scope', 'probe'), ('scope', 'language', 'family', 'mode')):
                key = tuple((axis, labels[axis]) for axis in axes)
                grouped.setdefault(key, []).append((row, scores[case['id']], records[case['id']]))
    groups = []
    for key, members in sorted(grouped.items()):
        objective = {}
        for name in sorted({name for _, score, _ in members for name in score['objective']}):
            values = [score['objective'][name]['passed'] for _, score, _ in members if name in score['objective']]
            objective[name] = {'expected': len(values), 'passed': sum(value is True for value in values)}
        diagnostics = {}
        for name in ('correction_uptake_exact', 'constraint_change_exact'):
            values = [record['controlled'][name] for _, _, record in members if record['controlled'][name] is not None]
            diagnostics[name] = {'expected': len(values), 'passed': sum(value is True for value in values)}
        judgments = {}
        for dimension in sorted({j['dimension'] for row, _, _ in members for j in row['judgments']}):
            values = [j['label'] for row, _, _ in members for j in row['judgments'] if j['dimension'] == dimension]
            judgments[dimension] = {label: values.count(label) for label in LABELS}
        groups.append({'labels': dict(key), 'expected_rows': len(members),
                       'source_groups': len({row['case']['source_group_id'] for row, _, _ in members}),
                       'states': {state: sum(row['response']['state'] == state for row, _, _ in members) for state in metrics.STATES},
                       'objective': objective, 'controlled_changes': diagnostics,
                       'whole_answer_agent_labels': {label: sum(record['whole_answer_label'] == label for _, _, record in members)
                                                     for label in LABELS},
                       'native_review_required': sum(record['native_review_required'] for _, _, record in members),
                       'agent_judgments': judgments})
    return {'version': REVIEW_VERSION, 'groups': groups, 'model_judgments_are_gold': False,
            'semantic_quality_established': False, 'clinical_certification': False,
            'scope_limit': 'Derived response views share source groups; no independent-scenario or native-gold claim.'}


def expected_cases(evaluation, worker):
    """Reconstruct remote-slot mapping from frozen cases, not collected result order."""
    cases, mapping = {}, []
    for scope, name in (('fresh', 'cases.jsonl'), ('reused', 'reused-regressions.jsonl')):
        for case in inputs.json_lines(evaluation / name):
            metrics.require(case['id'] not in cases, 'Duplicate frozen case ID')
            cases[case['id']] = case
            mapping.append({'id': case['id'], 'case_id': case['id'], 'scope': scope, 'kind': 'fixed',
                            'source_group_id': case['source_group_id'], 'language': case['input_language'],
                            'probe': case['probe']})
    for scenario in inputs.json_lines(evaluation / 'scenarios.jsonl'):
        for turn, case in enumerate(scenario['expected_per_turn']):
            metrics.require(case['id'] not in cases, 'Duplicate frozen rollout case ID')
            cases[case['id']] = case
            mapping.append({'id': worker.rollout_slot_id(scenario['id'], turn), 'case_id': case['id'],
                            'scope': 'fresh', 'kind': 'rollout', 'source_group_id': case['source_group_id'],
                            'language': case['input_language'], 'probe': case['probe']})
    return cases, mapping


def convert(package, execution_sha, progress_path, expected_snapshot_sha, *, evaluation=None,
            annotations=None, expected_annotations_sha=None):
    worker, execution, plan, slots = launcher.verify_package(package, execution_sha)
    evaluation = Path(evaluation or execution['evaluation']).resolve()
    evaluation_sha = execution['binding']['evaluation_manifest_sha256']
    verified_inputs = inputs.verify(evaluation, expected_manifest_sha256=evaluation_sha)
    cases, expected_mapping = expected_cases(evaluation, worker)
    metrics.require(slots == expected_mapping, 'Prepared slot mapping differs from frozen evaluation cases')
    raw_bytes = Path(progress_path).read_bytes()
    progress = worker.parse(raw_bytes)
    metrics.require(metrics.is_sha(expected_snapshot_sha) and progress.get('snapshot_sha256') == expected_snapshot_sha,
                    'Expected independent progress snapshot hash differs')
    worker.verify_report(progress, plan, execution['binding'])
    manifest = worker.verify_bundle(execution['bundle'], worker.EXPECTED)
    identity = {'model': manifest['model'], 'revision': manifest['revision'], 'base_weights_sha256': manifest['base_sha256'],
                'adapter_weights_sha256': manifest['variants']['r6_response']['files']['adapter_model.safetensors']}
    done = {record['id']: record for record in progress['results']}
    rows = {scope: [] for scope in SCOPES}
    row_sources = []
    for slot in slots:
        case = deepcopy(cases[slot['case_id']])
        record = done.get(slot['id'])
        state = record['state'] if record is not None else 'pending'
        if record is not None:
            messages, output = deepcopy(record['messages']), record['output']
        else:
            messages = deepcopy(case['messages']) if case['mode'] == 'fixed_shared_history' or case['turn'] == 0 else []
            output = ''
        row = {'version': metrics.VERSION, 'case': case, 'model_id': 'r6_response',
               'model_sha256': metrics.digest(identity), 'protocol_sha256': execution['binding']['protocol_sha256'],
               'messages': messages,
               'response': {'state': state, 'output': output, 'disposition': 'unknown', 'executions': [], 'confirmations': []},
               'judgments': []}
        rows[slot['scope']].append(row)
        row_sources.append({'case_id': case['id'], 'remote_slot_id': slot['id'], 'scope': slot['scope'],
                            'state': state, 'record_present': record is not None,
                            'worker_evidence_verified': bool(record and record['verified']),
                            'raw_result_sha256': metrics.digest(record) if record is not None else None})
    expected_counts = {'fresh': verified_inputs['counts']['fresh_derived_test_views'],
                       'reused': verified_inputs['counts']['reused_fixed_views']}
    metrics.require({scope: len(value) for scope, value in rows.items()} == expected_counts
                    and sum(expected_counts.values()) == execution['counts']['expected_slots'], 'Expected scope denominators differ')
    review, decisions = None, None
    if annotations is not None:
        metrics.require(metrics.is_sha(expected_annotations_sha) and file_sha(annotations) == expected_annotations_sha,
                        'Independent annotation file hash differs')
        decisions = inputs.read(annotations)
        review = bind_review(rows, row_sources, progress, decisions, execution_sha)
    else:
        metrics.require(expected_annotations_sha is None, 'Annotation hash without annotation file')
    reports, scope_coverage = {}, {}
    for scope, scoped_rows in rows.items():
        reports[scope] = metrics.evaluate_rows(scoped_rows, expected_rows_sha256=metrics.digest(scoped_rows))
        scope_coverage[scope] = coverage(scoped_rows)
    metrics.require({state: sum(scope_coverage[scope]['states'][state] for scope in SCOPES) for state in metrics.STATES}
                    == progress['counts']['states'], 'Converted state counts differ from raw checkpoint')
    summary = {'version': VERSION, 'run_id': execution['binding']['run_id'], 'raw_progress_state': progress['state'],
               'execution_sha256': execution_sha, 'evaluation_manifest_sha256': evaluation_sha,
               'progress_snapshot_sha256': expected_snapshot_sha, 'raw_progress_file_sha256': hashlib.sha256(raw_bytes).hexdigest(),
               'model_identity': identity, 'counts': progress['counts'], 'scopes': scope_coverage,
               'evidence': {'frozen_input_bindings_checked': True, 'worker_report_verified_slots': progress['counts']['verified_slots'],
                            'independent_native_reexecution': False, 'quality_inferred_from_evidence_verification': False},
               'disposition_policy': 'unknown_without_attributed_annotation_not_inferred_from_prose',
               'semantic_judgments': 'attributed_agent_not_gold' if review else 'not_supplied',
               'annotations_sha256': expected_annotations_sha, 'semantic_quality_established': False,
               'clinical_certification': False, 'model_judgments_are_gold': False,
               'scope_limit': 'AI-authored development and separate REUSED regressions; not untouched final-test or native gold.'}
    return {'rows': rows, 'reports': reports, 'summary': summary, 'row_sources': row_sources, 'raw_bytes': raw_bytes,
            'decisions': decisions, 'review': review, 'review_summary': review_summary(rows, reports, review) if review else None}


def score(package, execution_sha, progress_path, expected_snapshot_sha, out, *, evaluation=None,
          annotations=None, expected_annotations_sha=None):
    out = Path(out)
    metrics.require(not out.exists(), 'Use a new score directory; do not overwrite prior snapshots')
    converted = convert(package, execution_sha, progress_path, expected_snapshot_sha, evaluation=evaluation,
                        annotations=annotations, expected_annotations_sha=expected_annotations_sha)
    _, execution, _, _ = launcher.verify_package(package, execution_sha)
    metrics.require(not out.resolve().is_relative_to(Path(package).resolve())
                    and not out.resolve().is_relative_to(Path(evaluation or execution['evaluation']).resolve()),
                    'Do not write inside frozen input packages')
    payloads = {'raw-progress.json': converted['raw_bytes'], 'summary.json': inputs.encoded(converted['summary']),
                'row-sources.json': inputs.encoded(converted['row_sources'])}
    if converted['review'] is not None:
        payloads.update({'annotations.json': Path(annotations).read_bytes(),
                         'agent-review.json': inputs.encoded(converted['review']),
                         'review-summary.json': inputs.encoded(converted['review_summary'])})
    for scope in SCOPES:
        payloads[scope + '.rows.json'] = inputs.encoded(converted['rows'][scope])
        payloads[scope + '.metrics.json'] = inputs.encoded(converted['reports'][scope])
    manifest = {'version': VERSION, 'private': True,
                'execution_sha256': execution_sha, 'evaluation_manifest_sha256': converted['summary']['evaluation_manifest_sha256'],
                'progress_snapshot_sha256': expected_snapshot_sha,
                'artifacts': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())},
                'code': {name: file_sha(Path(__file__).with_name(name)) for name in (
                    'score_r7_behavior_baseline.py', 'r7_eval_metrics.py', 'build_r7_behavior_eval.py', 'run_r7_behavior_baseline.py')},
                'scope_counts': {scope: len(converted['rows'][scope]) for scope in SCOPES},
                'semantic_quality_established': False, 'clinical_certification': False, 'model_judgments_are_gold': False}
    payloads['manifest.json'] = inputs.encoded(manifest)
    checksum = hashlib.sha256(payloads['manifest.json']).hexdigest()
    payloads['FROZEN.json'] = inputs.encoded({'manifest_sha256': checksum})
    out.mkdir(parents=True)
    for name, data in payloads.items():
        with (out / name).open('xb') as stream:
            stream.write(data)
    return verify(out, expected_manifest_sha256=checksum)


def verify(out, *, expected_manifest_sha256):
    out = Path(out)
    metrics.require(metrics.is_sha(expected_manifest_sha256) and file_sha(out / 'manifest.json') == expected_manifest_sha256,
                    'Score manifest hash mismatch')
    manifest = inputs.read(out / 'manifest.json')
    metrics.require(inputs.read(out / 'FROZEN.json') == {'manifest_sha256': expected_manifest_sha256}, 'Score seal mismatch')
    metrics.require(manifest['version'] == VERSION and manifest['private'] is True
                    and manifest['semantic_quality_established'] is False and manifest['clinical_certification'] is False
                    and manifest['model_judgments_are_gold'] is False, 'Invalid scoring scope')
    metrics.require({p.name for p in out.iterdir()} == set(manifest['artifacts']) | {'manifest.json', 'FROZEN.json'},
                    'Score artifact inventory changed')
    for name, checksum in manifest['artifacts'].items():
        metrics.require(file_sha(inputs.safe_file(out, name)) == checksum, 'Score artifact changed: ' + name)
    for name, checksum in manifest['code'].items():
        metrics.require(Path(name).name == name and file_sha(Path(__file__).with_name(name)) == checksum, 'Scoring code changed')
    summary = inputs.read(out / 'summary.json')
    metrics.require(summary['semantic_quality_established'] is False and summary['clinical_certification'] is False
                    and summary['model_judgments_are_gold'] is False
                    and summary['evidence']['quality_inferred_from_evidence_verification'] is False, 'Unsupported quality claim')
    raw = inputs.read(out / 'raw-progress.json')
    metrics.require(raw['snapshot_sha256'] == manifest['progress_snapshot_sha256']
                    == metrics.digest({k: v for k, v in raw.items() if k != 'snapshot_sha256'})
                    and summary['counts'] == raw['counts'], 'Raw checkpoint/count binding differs')
    all_rows = {scope: inputs.read(out / (scope + '.rows.json')) for scope in SCOPES}
    all_reports = {}
    if summary.get('annotations_sha256') is not None:
        metrics.require(file_sha(out / 'annotations.json') == summary['annotations_sha256'], 'Annotation seal differs')
        blank = deepcopy(all_rows)
        for rows in blank.values():
            for row in rows:
                row['judgments'] = []
        review = bind_review(blank, inputs.read(out / 'row-sources.json'), raw,
                             inputs.read(out / 'annotations.json'), summary['execution_sha256'])
        metrics.require(blank == all_rows and review == inputs.read(out / 'agent-review.json'), 'Attributed review differs')
    else:
        review = None
        metrics.require(all(row['judgments'] == [] for rows in all_rows.values() for row in rows), 'Unattributed judgments')
    for scope in SCOPES:
        rows = all_rows[scope]
        result = metrics.evaluate_rows(rows, expected_rows_sha256=summary['scopes'][scope]['rows_sha256'])
        all_reports[scope] = result
        metrics.require(result == inputs.read(out / (scope + '.metrics.json'))
                        and coverage(rows) == summary['scopes'][scope]
                        and len(rows) == manifest['scope_counts'][scope] == summary['scopes'][scope]['expected_rows'],
                        'Metrics or scope denominator differs on recomputation')
    if review is not None:
        metrics.require(review_summary(all_rows, all_reports, review) == inputs.read(out / 'review-summary.json'),
                        'Review summary differs on recomputation')
    return {'manifest_sha256': expected_manifest_sha256, 'scopes': summary['scopes'],
            'raw_progress_state': summary['raw_progress_state'], 'evidence_bindings_verified': True,
            'semantic_quality_established': False, 'clinical_certification': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--expected-execution-sha')
    parser.add_argument('--progress', type=Path)
    parser.add_argument('--expected-snapshot-sha')
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--expected-annotations-sha')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--expected-manifest-sha')
    args = parser.parse_args(argv)
    if args.verify:
        result = verify(args.out, expected_manifest_sha256=args.expected_manifest_sha)
    else:
        if not all((args.package, args.expected_execution_sha, args.progress, args.expected_snapshot_sha)):
            parser.error('Scoring requires the package/execution seal and progress/snapshot seal')
        result = score(args.package, args.expected_execution_sha, args.progress, args.expected_snapshot_sha,
                       args.out, evaluation=args.evaluation, annotations=args.annotations,
                       expected_annotations_sha=args.expected_annotations_sha)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
