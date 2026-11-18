#!/usr/bin/env python3
"""Offline R7 diagnostics. No inference, network access, or clinical certification."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata
from typing import Literal, NotRequired, TypeAlias, TypedDict

Json: TypeAlias = None | bool | int | float | str | list['Json'] | dict[str, 'Json']
Mode = Literal['fixed_shared_history', 'independent_rollout']
State = Literal['complete', 'invalid', 'error', 'incomplete', 'blocked', 'pending']
Disposition = Literal['answer', 'abstain', 'refuse', 'unknown']
VERSION = 'r7-eval-v1'
STATES = ('complete', 'invalid', 'error', 'incomplete', 'blocked', 'pending')
DISPOSITIONS = ('answer', 'abstain', 'refuse', 'unknown')
DIMENSIONS = ('meaning', 'multiturn_usefulness', 'source_grounding', 'citation_support',
              'tool_appropriateness', 'commerce_integrity', 'language_adherence',
              'english_retention', 'abstention_appropriateness', 'refusal_appropriateness',
              'medical_safety', 'category_membership', 'list_distinctness', 'correction_uptake', 'topic_change')
LIST_CHECKS = ('category_membership', 'list_distinctness', 'correction_uptake', 'topic_change')
CHECKS = ('exact_text', 'json_equal', 'json_field_equal', 'citation_ids', 'tool_selection', 'tool_arguments',
          'tool_execution', 'commerce_confirmation', 'transaction_claims', 'disposition') + LIST_CHECKS


class Message(TypedDict):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str


class Constraint(TypedDict):
    id: str
    kind: str
    expected: Json


class EvalCase(TypedDict):
    id: str
    scenario_id: str
    source_group_id: str
    domain: str
    input_language: str
    output_language: str
    mode: Mode
    turn: int
    messages: list[Message]
    sources: dict[str, str]
    constraints: list[Constraint]
    rubrics: dict[str, str]
    probe: NotRequired[Literal['fresh', 'correction', 'followup', 'topic_change', 'unspecified']]


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict[str, Json]


class Execution(TypedDict):
    call: ToolCall
    sequence: int
    status: Literal['success', 'error']
    transaction_id: str | None
    confirmation_id: str | None


class Confirmation(TypedDict):
    id: str
    sequence: int
    accepted: bool
    name: str
    arguments: dict[str, Json]


class Response(TypedDict):
    state: State
    output: str
    disposition: Disposition
    executions: list[Execution]
    confirmations: list[Confirmation]


class Judge(TypedDict):
    kind: Literal['human', 'model']
    id: str
    version: str
    input: dict[str, Json]
    input_sha256: str


class Judgment(TypedDict):
    dimension: str
    label: Literal['pass', 'fail', 'abstain', 'unknown']
    judge: Judge


class EvalRow(TypedDict):
    version: str
    case: EvalCase
    model_id: str
    model_sha256: str
    protocol_sha256: str
    messages: list[Message]
    response: Response
    judgments: list[Judgment]


class Metric(TypedDict):
    kind: Literal['objective', 'human', 'model']
    name: str
    judge_id: NotRequired[str]
    judge_version: NotRequired[str]


class CheckResult(TypedDict):
    passed: bool
    observed: bool
    reason: str
    diagnostics: NotRequired[dict[str, Json]]


class RowResult(TypedDict):
    case_id: str
    model_id: str
    state: State
    disposition: Disposition
    objective: dict[str, CheckResult]
    repetition: dict[str, Json]
    structured_output_invalid: bool
    judgments: list[Judgment]


class EvaluationResult(TypedDict):
    version: str
    rows_sha256: str
    row_count: int
    unique_cases: int
    source_groups: int
    rows: list[RowResult]
    strata: list[dict[str, Json]]
    semantic_quality_established: bool
    clinical_certification: bool
    model_judgments_are_gold: bool


class SourceRecord(TypedDict):
    id: str
    source_group_id: str
    text: str


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def fields(value: object, names: str, optional: str = '') -> None:
    require(type(value) is dict and set(names.split()) <= set(value) <= set((names + ' ' + optional).split()),
            'Unexpected or missing fields: ' + names)


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _json_value(value: object) -> None:
    if type(value) is dict:
        require(all(type(k) is str for k in value), 'JSON keys must be strings')
        for item in value.values():
            _json_value(item)
    elif type(value) is list:
        for item in value:
            _json_value(item)
    else:
        require(type(value) in (type(None), bool, int, float, str), 'Not a JSON value')
        require(type(value) is not float or math.isfinite(value), 'Nonfinite JSON')


def digest(value: object) -> str:
    """Canonical JSON SHA256; callers must obtain expected hashes independently."""
    _json_value(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parse_json(text: str) -> Json:
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(text, object_pairs_hook=unique,
                        parse_constant=lambda _: require(False, 'Nonfinite JSON'))
    _json_value(result)
    return result


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def repetition_metrics(output: str, messages: list[Message], *, n: int = 4) -> dict[str, Json]:
    require(type(n) is int and n > 0, 'Positive n-gram size required')
    words = re.findall(r'\w+', normalized(output), flags=re.UNICODE)
    grams = Counter(tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1)))
    positions = sum(grams.values())
    repeats = sum(count - 1 for count in grams.values())
    key = normalized(output)
    previous = [normalized(m['content']) for m in messages if m['role'] == 'assistant']
    consecutive = int(bool(key))
    for prior in reversed(previous):
        if not key or prior != key:
            break
        consecutive += 1
    return {'word_count': len(words), 'ngram_size': n, 'ngram_positions': positions,
            'repeated_positions': repeats, 'repeated_fraction': repeats / positions if positions else None,
            'matches_prior_assistant': bool(key and key in previous),
            'same_as_previous_assistant': bool(key and previous and key == previous[-1]),
            'consecutive_identical_assistant_replies': consecutive}


def judgment_input(row: EvalRow, dimension: str) -> dict[str, Json]:
    """Exact review payload, excluding labels and model identity to permit blinding."""
    return {'dimension': dimension, 'case': row['case'], 'messages': row['messages'],
            'response': row['response']}


def _strings(value: object) -> bool:
    return (type(value) is list and all(nonempty(v) for v in value)
            and len(value) == len(set(value)))


def _messages(messages: object) -> None:
    require(type(messages) is list, 'Messages must be a list')
    for message in messages:
        fields(message, 'role content')
        require(message['role'] in ('system', 'user', 'assistant', 'tool')
                and isinstance(message['content'], str), 'Invalid message')


def _calls(calls: object) -> None:
    require(type(calls) is list, 'Tool calls must be a list')
    ids = set()
    for call in calls:
        fields(call, 'id name arguments')
        require(nonempty(call['id']) and nonempty(call['name'])
                and type(call['arguments']) is dict, 'Invalid tool call')
        require(call['id'] not in ids, 'Duplicate tool call ID')
        ids.add(call['id'])


def _constraints(case: EvalCase) -> None:
    require(type(case['constraints']) is list, 'Constraints must be a list')
    ids = set()
    for check in case['constraints']:
        fields(check, 'id kind expected')
        name, kind, expected = check['id'], check['kind'], check['expected']
        require(nonempty(name) and name not in ids and kind in CHECKS, 'Invalid/duplicate constraint')
        ids.add(name)
        if kind == 'exact_text':
            require(isinstance(expected, str), 'Expected text required')
        elif kind == 'json_field_equal':
            fields(expected, 'field value')
            require(nonempty(expected['field']), 'Expected JSON field name required')
        elif kind == 'citation_ids':
            fields(expected, 'allowed required')
            require(_strings(expected['allowed']) and _strings(expected['required'])
                    and set(expected['required']) <= set(expected['allowed']) <= set(case['sources']),
                    'Citation constraints require registered sources')
        elif kind in ('tool_selection', 'commerce_confirmation'):
            require(type(expected) is list and all(nonempty(v) for v in expected), 'Expected tool names required')
        elif kind == 'tool_arguments':
            _calls(expected)
        elif kind in ('tool_execution', 'transaction_claims'):
            require(expected is None, 'Trace checks take null expected')
        elif kind == 'disposition':
            require(expected in DISPOSITIONS, 'Unknown expected disposition')
        elif kind in LIST_CHECKS:
            fields(expected, 'format field count allowed forbidden aliases')
            require(expected['format'] in ('json_field', 'numbered_lines')
                    and ((expected['format'] == 'json_field' and nonempty(expected['field']))
                         or (expected['format'] == 'numbered_lines' and expected['field'] is None)),
                    'Invalid frozen list parser')
            require(type(expected['count']) is int and expected['count'] > 0
                    and _strings(expected['allowed']) and _strings(expected['forbidden']), 'Invalid list expectations')
            for key in ('allowed', 'forbidden'):
                require(all(normalized(v) == v for v in expected[key]), 'Use normalized canonical item IDs')
            require(not set(expected['allowed']) & set(expected['forbidden']), 'Conflicting list expectations')
            aliases = expected['aliases']
            require(type(aliases) is dict and all(nonempty(k) and normalized(k) == k
                    and v in expected['allowed'] + expected['forbidden'] for k, v in aliases.items()),
                    'Aliases must map normalized text to frozen canonical items')
            require(all(k == v or k not in expected['allowed'] + expected['forbidden'] for k, v in aliases.items()),
                    'Alias cannot redefine a canonical item')


def _response(response: Response) -> None:
    fields(response, 'state output disposition executions confirmations')
    require(response['state'] in STATES and response['disposition'] in DISPOSITIONS
            and isinstance(response['output'], str), 'Invalid response state/output/disposition')
    require(response['state'] != 'complete' or nonempty(response['output']), 'Empty complete output')
    require(type(response['executions']) is list and type(response['confirmations']) is list,
            'Execution and confirmation evidence must be lists')
    _calls([execution['call'] for execution in response['executions'] if type(execution) is dict and 'call' in execution])
    sequences, confirmation_ids = set(), set()
    for event in response['executions'] + response['confirmations']:
        require(type(event) is dict, 'Trace events must be objects')
        is_execution = 'call' in event
        fields(event, 'call sequence status transaction_id confirmation_id' if is_execution
               else 'id sequence accepted name arguments')
        sequence = event['sequence']
        require(type(sequence) is int and sequence >= 0 and sequence not in sequences,
                'Event sequence must be unique nonnegative integers')
        sequences.add(sequence)
        if is_execution:
            require(event['status'] in ('success', 'error'), 'Unknown execution status')
            require(all(event[k] is None or nonempty(event[k]) for k in ('transaction_id', 'confirmation_id')),
                    'Invalid receipt IDs')
            require(event['status'] == 'success' or event['transaction_id'] is None,
                    'Failed execution cannot have a transaction ID')
        else:
            require(nonempty(event['id']) and event['id'] not in confirmation_ids
                    and type(event['accepted']) is bool and nonempty(event['name'])
                    and type(event['arguments']) is dict, 'Invalid/duplicate confirmation')
            confirmation_ids.add(event['id'])


def validate_rows(rows: list[EvalRow], *, expected_rows_sha256: str) -> None:
    require(is_sha(expected_rows_sha256) and digest(rows) == expected_rows_sha256, 'Rows hash mismatch')
    require(type(rows) is list, 'Rows must be a list')
    seen, cases, identities, protocols = set(), {}, {}, set()
    scenario_groups, constraint_kinds = {}, {}
    rollouts = defaultdict(list)
    for row in rows:
        fields(row, 'version case model_id model_sha256 protocol_sha256 messages response judgments')
        require(row['version'] == VERSION and nonempty(row['model_id'])
                and is_sha(row['model_sha256']) and is_sha(row['protocol_sha256']), 'Invalid row identity')
        require(identities.setdefault(row['model_id'], row['model_sha256']) == row['model_sha256'],
                'Model identity changed')
        protocols.add(row['protocol_sha256'])
        case = row['case']
        fields(case, 'id scenario_id source_group_id domain input_language output_language mode turn messages sources constraints rubrics', 'probe')
        require(case.get('probe', 'unspecified') in ('fresh', 'correction', 'followup', 'topic_change', 'unspecified'),
                'Unknown probe taxonomy')
        require(all(nonempty(case[k]) for k in ('id', 'scenario_id', 'source_group_id', 'domain',
                                              'input_language', 'output_language')), 'Missing case metadata')
        require(case['mode'] in ('fixed_shared_history', 'independent_rollout')
                and type(case['turn']) is int and case['turn'] >= 0, 'Invalid mode/turn')
        key = (row['model_id'], case['id'])
        require(key not in seen, 'Duplicate model/case row')
        seen.add(key)
        require(cases.setdefault(case['id'], digest(case)) == digest(case), 'Case changed across models')
        _messages(case['messages'])
        _messages(row['messages'])
        require(bool(case['messages']), 'Case messages required')
        require(type(case['sources']) is dict and all(nonempty(k) and nonempty(v)
                for k, v in case['sources'].items()), 'Invalid sources')
        require(type(case['rubrics']) is dict and set(case['rubrics']) <= set(DIMENSIONS)
                and all(nonempty(v) for v in case['rubrics'].values()), 'Invalid semantic rubrics')
        _constraints(case)
        scenario = (case['mode'], case['scenario_id'])
        require(scenario_groups.setdefault(scenario, case['source_group_id']) == case['source_group_id'],
                'Scenario source group changed')
        for check in case['constraints']:
            require(constraint_kinds.setdefault(check['id'], check['kind']) == check['kind'],
                    'Constraint ID changes metric kind')
        _response(row['response'])
        if case['mode'] == 'fixed_shared_history':
            require(row['messages'] == case['messages'], 'Fixed shared history changed')
        else:
            require(all(m['role'] in ('user', 'system') for m in case['messages']),
                    'Independent stimuli must contain only system/user messages')
            rollouts[(row['model_id'], case['scenario_id'])].append(row)
        require(type(row['judgments']) is list, 'Judgments must be a list')
        judges = set()
        for judgment in row['judgments']:
            fields(judgment, 'dimension label judge')
            dimension, judge = judgment['dimension'], judgment['judge']
            require(nonempty(dimension) and dimension in case['rubrics']
                    and judgment['label'] in ('pass', 'fail', 'abstain', 'unknown'),
                    'Unknown judgment dimension/label')
            fields(judge, 'kind id version input input_sha256')
            require(judge['kind'] in ('human', 'model') and nonempty(judge['id']) and nonempty(judge['version']),
                    'Judge kind, model/reviewer ID and version required')
            identity = (dimension, judge['kind'], judge['id'], judge['version'])
            require(identity not in judges, 'Duplicate judgment')
            judges.add(identity)
            require(is_sha(judge['input_sha256']) and digest(judge['input']) == judge['input_sha256']
                    == digest(judgment_input(row, dimension)), 'Judge input mismatch')
    require(len(protocols) <= 1, 'Different protocols require separate reports')
    for trajectory in rollouts.values():
        trajectory.sort(key=lambda r: r['case']['turn'])
        require([r['case']['turn'] for r in trajectory] == list(range(len(trajectory))),
                'Independent rollout turns must be contiguous, including pending slots')
        previous = None
        for row in trajectory:
            case, messages = row['case'], row['messages']
            require(case['source_group_id'] == trajectory[0]['case']['source_group_id'],
                    'Rollout source group changed')
            if previous is None:
                expected = case['messages']
            else:
                prior_stimuli = previous['case']['messages']
                require(case['messages'][:len(prior_stimuli)] == prior_stimuli
                        and len(case['messages']) > len(prior_stimuli), 'Rollout stimuli changed')
                expected = previous['messages'] + [{'role': 'assistant', 'content': previous['response']['output']}]
                expected += case['messages'][len(prior_stimuli):]
            if messages:
                require(previous is None or previous['response']['state'] == 'complete',
                        'Cannot continue an unsuccessful rollout turn')
                require(messages == expected, 'Independent history contains another or missing response')
            else:
                require(row['response']['state'] in ('pending', 'blocked', 'error'), 'Missing rollout history')
            previous = row


def _structured(output: str) -> dict[str, Json] | None:
    try:
        parsed = parse_json(output)
        return parsed if type(parsed) is dict else None
    except (ValueError, RecursionError):
        return None


def _list_items(output: str, expected: dict[str, Json]) -> list[str] | None:
    if expected['format'] == 'json_field':
        payload = _structured(output)
        items = payload.get(expected['field']) if payload is not None else None
        if type(items) is not list or not all(nonempty(v) for v in items):
            return None
    else:
        matches = re.findall(r'^\s*([0-9]+)[.)]\s+(.+?)\s*$', output, re.MULTILINE)
        if not matches or [int(number) for number, _ in matches] != list(range(1, len(matches) + 1)):
            return None
        items = [text for _, text in matches]
    return [expected['aliases'].get(normalized(item), normalized(item)) for item in items]


def _list_diagnostics(check: Constraint, row: EvalRow) -> dict[str, Json]:
    expected = check['expected']
    items = _list_items(row['response']['output'], expected) if row['response']['state'] == 'complete' else None
    prior = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
    previous = _list_items(prior[-1], expected) if prior else None
    actual = items or []
    valid = [item for item in actual if item in expected['allowed']]
    obsolete = sum(item in expected['forbidden'] for item in actual)
    return {'parsed': items is not None, 'expected_items': expected['count'], 'returned_items': len(actual),
            'distinct_items': len(set(actual)), 'duplicate_items': len(actual) - len(set(actual)),
            'in_category_items': len(valid), 'unlisted_items': len(actual) - len(valid),
            'distinct_in_category_items': len(set(valid)), 'obsolete_items': obsolete,
            'membership_fraction_returned': _rate(len(valid), len(actual)),
            'distinctness_fraction_returned': _rate(len(set(actual)), len(actual)),
            'obsolete_fraction_returned': _rate(obsolete, len(actual)),
            'prior_list_available': previous is not None,
            'prior_out_of_category_items': sum(item not in expected['allowed'] for item in previous)
                if previous is not None else None}


def _check(check: Constraint, row: EvalRow, payload: dict[str, Json] | None) -> bool:
    kind, expected, response = check['kind'], check['expected'], row['response']
    if kind in LIST_CHECKS:
        diagnostic = _list_diagnostics(check, row)
        count_matches = diagnostic['parsed'] and diagnostic['returned_items'] == expected['count']
        membership = count_matches and diagnostic['unlisted_items'] == 0
        distinct = count_matches and diagnostic['distinct_items'] == expected['count']
        if kind == 'category_membership':
            return membership
        if kind == 'list_distinctness':
            return distinct
        return bool(membership and distinct and not diagnostic['obsolete_items']
                    and any(m['role'] == 'assistant' for m in row['messages']))
    if kind == 'exact_text':
        return response['output'] == expected
    if kind == 'json_equal':
        try:
            return digest(parse_json(response['output'])) == digest(expected)
        except (ValueError, RecursionError):
            return False
    if kind == 'disposition':
        return response['disposition'] == expected
    if kind == 'json_field_equal':
        return (payload is not None and expected['field'] in payload
                and digest(payload[expected['field']]) == digest(expected['value']))
    if kind == 'commerce_confirmation':
        confirmations = {c['id']: c for c in response['confirmations']}
        used = set()
        for execution in response['executions']:
            call = execution['call']
            if call['name'] not in expected:
                continue
            confirmation = confirmations.get(execution['confirmation_id'])
            if not (confirmation and confirmation['accepted'] and confirmation['sequence'] < execution['sequence']
                    and confirmation['name'] == call['name'] and digest(confirmation['arguments']) == digest(call['arguments'])
                    and confirmation['id'] not in used):
                return False
            used.add(confirmation['id'])
        return True
    if payload is None:
        return False
    if kind == 'citation_ids':
        cited = payload.get('citations')
        return (_strings(cited) and set(expected['required']) <= set(cited) <= set(expected['allowed']))
    if kind == 'transaction_claims':
        claims = payload.get('transaction_claims')
        if type(claims) is not list:
            return False
        successful = {(e['call']['id'], e['transaction_id']) for e in response['executions']
                      if e['status'] == 'success' and e['transaction_id'] is not None}
        seen = set()
        for claim in claims:
            if (type(claim) is not dict or set(claim) != {'call_id', 'transaction_id'}
                    or not all(nonempty(v) for v in claim.values())):
                return False
            identity = (claim['call_id'], claim['transaction_id'])
            if identity not in successful or identity in seen:
                return False
            seen.add(identity)
        return True
    calls = payload.get('tool_calls')
    try:
        _calls(calls)
    except ValueError:
        return False
    if kind == 'tool_selection':
        return [call['name'] for call in calls] == expected
    if kind == 'tool_arguments':
        return digest(calls) == digest(expected)
    executions = sorted(response['executions'], key=lambda e: e['sequence'])
    return (digest(calls) == digest([e['call'] for e in executions])
            and all(e['status'] == 'success' for e in executions))


def _score(row: EvalRow) -> RowResult:
    response = row['response']
    payload = _structured(response['output'])
    objective = {}
    for check in row['case']['constraints']:
        observed = response['state'] == 'complete'
        passed = observed and _check(check, row, payload)
        objective[check['id']] = {'passed': passed, 'observed': observed,
                                 'reason': 'pass' if passed else 'mismatch' if observed else response['state']}
        if check['kind'] in LIST_CHECKS:
            objective[check['id']]['diagnostics'] = _list_diagnostics(check, row)
    needs_object = any(c['kind'] in ('json_field_equal', 'citation_ids', 'tool_selection', 'tool_arguments', 'tool_execution',
                                    'transaction_claims') for c in row['case']['constraints'])
    return {'case_id': row['case']['id'], 'model_id': row['model_id'], 'state': response['state'],
            'disposition': response['disposition'], 'objective': objective,
            'repetition': repetition_metrics(response['output'], row['messages']),
            'structured_output_invalid': bool(needs_object and payload is None and response['output']),
            'judgments': row['judgments']}


def _rate(passes: int, total: int) -> float | None:
    return passes / total if total else None


def _counts(rows: list[EvalRow], scores: list[RowResult]) -> dict[str, Json]:
    states = Counter(r['response']['state'] for r in rows)
    dispositions = Counter(r['response']['disposition'] for r in rows)
    objectives = {}
    for name in sorted({name for score in scores for name in score['objective']}):
        checks = [score['objective'][name] for score in scores if name in score['objective']]
        passed, observed = sum(c['passed'] for c in checks), sum(c['observed'] for c in checks)
        objectives[name] = {'expected': len(checks), 'observed': observed, 'unavailable': len(checks) - observed,
                            'passed': passed, 'failed_observed': observed - passed,
                            'pass_rate_all_expected': _rate(passed, len(checks))}
        diagnostics = [check['diagnostics'] for check in checks if 'diagnostics' in check]
        if diagnostics:
            counts = {key: sum(d[key] for d in diagnostics) for key in (
                'expected_items', 'returned_items', 'distinct_items', 'duplicate_items',
                'in_category_items', 'unlisted_items', 'distinct_in_category_items', 'obsolete_items')}
            counts['parsed_rows'] = sum(d['parsed'] for d in diagnostics)
            counts['unparsed_rows'] = len(diagnostics) - counts['parsed_rows']
            counts['distinct_valid_coverage_all_requested'] = _rate(
                sum(min(d['expected_items'], d['distinct_in_category_items']) for d in diagnostics), counts['expected_items'])
            counts['prior_list_rows'] = sum(d['prior_list_available'] for d in diagnostics)
            counts['prior_out_of_category_rows'] = sum(bool(d['prior_out_of_category_items']) for d in diagnostics)
            objectives[name]['list_diagnostics'] = counts
    semantic = []
    judge_keys = sorted({(j['dimension'], j['judge']['kind'], j['judge']['id'], j['judge']['version'])
                        for row in rows for j in row['judgments']})
    for dimension, kind, judge_id, version in judge_keys:
        expected = sum(dimension in r['case']['rubrics'] for r in rows)
        labels = Counter(j['label'] for row in rows for j in row['judgments']
                         if (j['dimension'], j['judge']['kind'], j['judge']['id'], j['judge']['version'])
                         == (dimension, kind, judge_id, version))
        decided = labels['pass'] + labels['fail']
        semantic.append({'dimension': dimension, 'judge_kind': kind, 'judge_id': judge_id, 'judge_version': version,
                         'expected': expected, 'submitted': sum(labels.values()), 'missing': expected - sum(labels.values()),
                         'labels': {label: labels[label] for label in ('pass', 'fail', 'abstain', 'unknown')},
                         'decided': decided, 'pass_rate_decided_only': _rate(labels['pass'], decided),
                         'pass_rate_all_expected': _rate(labels['pass'], expected), 'is_gold': False})
    nonempty_scores = [score for row, score in zip(rows, scores) if normalized(row['response']['output'])]
    return {'expected_rows': len(rows), 'unique_cases': len({r['case']['id'] for r in rows}),
            'source_groups': len({r['case']['source_group_id'] for r in rows}),
            'states': {state: states[state] for state in STATES},
            'dispositions': {d: dispositions[d] for d in DISPOSITIONS}, 'disposition_denominator': len(rows),
            'structured_invalid_rows': sum(s['structured_output_invalid'] for s in scores),
            'objective': objectives,
            'semantic_expected': {d: sum(d in r['case']['rubrics'] for r in rows) for d in DIMENSIONS},
            'semantic': semantic,
            'repetition': {'all_rows': len(rows), 'nonempty_outputs': len(nonempty_scores),
                           'within_reply_flagged': sum(s['repetition']['repeated_positions'] > 0 for s in nonempty_scores),
                           'prior_reply_flagged': sum(s['repetition']['matches_prior_assistant'] for s in nonempty_scores)}}


def _strata(rows: list[EvalRow], *, include_model: bool):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        case = row['case']
        base = {'mode': case['mode']}
        if include_model:
            base['model_id'] = row['model_id']
        labels = {**base, **{k: case[k] for k in ('domain', 'input_language', 'output_language')},
                  'probe': case.get('probe', 'unspecified')}
        for label in (base, labels):
            groups[json.dumps(label, sort_keys=True)].append(index)
    return [(json.loads(label), indices) for label, indices in sorted(groups.items())]


def evaluate_rows(rows: list[EvalRow], *, expected_rows_sha256: str) -> EvaluationResult:
    validate_rows(rows, expected_rows_sha256=expected_rows_sha256)
    scores = [_score(row) for row in rows]
    strata = [{'labels': labels, **_counts([rows[i] for i in indices], [scores[i] for i in indices])}
              for labels, indices in _strata(rows, include_model=True)]
    return {'version': VERSION, 'rows_sha256': expected_rows_sha256, 'row_count': len(rows),
            'unique_cases': len({r['case']['id'] for r in rows}),
            'source_groups': len({r['case']['source_group_id'] for r in rows}), 'rows': scores, 'strata': strata,
            'semantic_quality_established': False, 'clinical_certification': False, 'model_judgments_are_gold': False}


def _metric(row: EvalRow, score: RowResult, metric: Metric) -> float | None:
    if metric['kind'] == 'objective':
        check = score['objective'].get(metric['name'])
        return float(check['passed']) if check is not None else None
    matches = [j for j in row['judgments'] if j['dimension'] == metric['name']
               and j['judge']['kind'] == metric['kind'] and j['judge']['id'] == metric['judge_id']
               and j['judge']['version'] == metric['judge_version']]
    if not matches or matches[0]['label'] not in ('pass', 'fail'):
        return None
    return float(matches[0]['label'] == 'pass')


def _quantile(values: list[float], p: float) -> float:
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _paired(rows: list[EvalRow], baseline: str, candidate: str, metric: Metric,
            seed: int, resamples: int) -> dict[str, Json]:
    by_model = {model: {r['case']['id']: r for r in rows if r['model_id'] == model}
                for model in (baseline, candidate)}
    before, after = by_model[baseline], by_model[candidate]
    common = sorted(before.keys() & after.keys())
    clusters = defaultdict(list)
    gains = losses = ties = missing_metric = 0
    unavailable_before = unavailable_after = 0
    for case_id in common:
        left, right = before[case_id], after[case_id]
        a, b = _metric(left, _score(left), metric), _metric(right, _score(right), metric)
        unavailable_before += left['response']['state'] != 'complete'
        unavailable_after += right['response']['state'] != 'complete'
        if a is None or b is None:
            missing_metric += 1
            continue
        delta = b - a
        gains += delta > 0
        losses += delta < 0
        ties += delta == 0
        clusters[left['case']['source_group_id']].append(delta)
    means = [sum(values) / len(values) for _, values in sorted(clusters.items())]
    interval = None
    if len(means) >= 2:
        rng = random.Random(seed)
        samples = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(resamples))
        interval = [_quantile(samples, 0.025), _quantile(samples, 0.975)]
    count = gains + losses + ties
    return {'baseline_rows': len(before), 'candidate_rows': len(after), 'matched_rows': len(common),
            'baseline_only': len(before.keys() - after.keys()), 'candidate_only': len(after.keys() - before.keys()),
            'missing_metric_pairs': missing_metric, 'scored_pairs': count,
            'unavailable_baseline_in_matched': unavailable_before, 'unavailable_candidate_in_matched': unavailable_after,
            'row_gains': gains, 'row_losses': losses, 'row_ties': ties,
            'row_weighted_delta': (gains - losses) / count if count else None,
            'groups': [{'source_group_id': group, 'pairs': len(values), 'delta': sum(values) / len(values)}
                       for group, values in sorted(clusters.items())],
            'group_count': len(means), 'group_gains': sum(v > 0 for v in means),
            'group_losses': sum(v < 0 for v in means), 'group_ties': sum(v == 0 for v in means),
            'group_macro_delta': sum(means) / len(means) if means else None,
            'uncertainty': {'method': 'paired_source_group_percentile_bootstrap', 'confidence': 0.95,
                            'estimand': 'equal_weight_source_group_mean_delta', 'interval': interval,
                            'resamples': resamples, 'seed': seed,
                            'unavailable_reason': None if interval is not None else 'fewer_than_two_scored_groups',
                            'multiple_comparison_adjustment': None}}


def paired_report(rows: list[EvalRow], *, expected_rows_sha256: str, baseline: str, candidate: str,
                  metric: Metric, seed: int = 17, resamples: int = 2000) -> dict[str, Json]:
    validate_rows(rows, expected_rows_sha256=expected_rows_sha256)
    require(nonempty(baseline) and nonempty(candidate) and baseline != candidate, 'Distinct models required')
    require(type(seed) is int and type(resamples) is int and resamples >= 100, 'At least 100 resamples required')
    require(type(metric) is dict and metric.get('kind') in ('objective', 'human', 'model'), 'Invalid metric')
    fields(metric, 'kind name' if metric['kind'] == 'objective' else 'kind name judge_id judge_version')
    require(all(nonempty(v) for v in metric.values()), 'Metric identity required')
    selected = [r for r in rows if r['model_id'] in (baseline, candidate)]
    return {'version': VERSION, 'rows_sha256': expected_rows_sha256, 'baseline': baseline, 'candidate': candidate,
            'metric': metric, 'model_judgments_are_gold': False, 'clinical_certification': False,
            'strata': [{'labels': labels, **_paired([selected[i] for i in indices], baseline, candidate, metric, seed, resamples)}
                       for labels, indices in _strata(selected, include_model=False)]}


def audit_overlap(training: list[SourceRecord], evaluation: list[SourceRecord], *,
                  expected_training_sha256: str, expected_evaluation_sha256: str) -> dict[str, Json]:
    """Exact/normalized/source-group audit of supplied inventories, not pretraining clearance."""
    for records, expected in ((training, expected_training_sha256), (evaluation, expected_evaluation_sha256)):
        require(is_sha(expected) and digest(records) == expected, 'Source inventory hash mismatch')
        require(type(records) is list, 'Source inventory must be a list')
        ids = set()
        for record in records:
            fields(record, 'id source_group_id text')
            require(all(nonempty(value) for value in record.values()) and record['id'] not in ids,
                    'Missing/duplicate source identity or text')
            ids.add(record['id'])
    checks = {}
    transforms = {'source_id': lambda r: r['id'], 'source_group': lambda r: r['source_group_id'],
                  'exact_text': lambda r: digest(r['text']), 'normalized_text': lambda r: digest(normalized(r['text']))}
    for name, transform in transforms.items():
        train_keys = {transform(r) for r in training}
        overlap = [r['id'] for r in evaluation if transform(r) in train_keys]
        checks[name] = {'overlapping_eval_ids': sorted(overlap), 'overlapping_eval_records': len(overlap)}
    return {'version': VERSION, 'training_sha256': expected_training_sha256,
            'evaluation_sha256': expected_evaluation_sha256, 'training_records': len(training),
            'evaluation_records': len(evaluation), 'checks': checks,
            'supplied_inventory_checks_disjoint': bool(training and evaluation)
                and all(c['overlapping_eval_records'] == 0 for c in checks.values()),
            'inventory_coverage_verified': False, 'near_duplicate_review': 'not_assessed',
            'pretraining_overlap': 'unknown', 'independent_holdout_established': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rows', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    rows = parse_json(args.rows.read_text(encoding='utf-8'))
    result = evaluate_rows(rows, expected_rows_sha256=args.expected_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
