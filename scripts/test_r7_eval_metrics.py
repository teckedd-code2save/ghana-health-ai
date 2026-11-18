"""Synthetic CPU-only contract tests; no evaluation case counts or quality claims."""
from copy import deepcopy
import json
import unittest

import r7_eval_metrics as metrics


def row(case_id='case-0', model='r6', group='source-0', output='ready', **case_updates):
    case = {'id': case_id, 'scenario_id': case_id, 'source_group_id': group,
            'domain': 'meaning', 'input_language': 'tw', 'output_language': 'tw',
            'mode': 'fixed_shared_history', 'turn': 0,
            'messages': [{'role': 'user', 'content': 'Synthetic fixture input'}], 'sources': {},
            'constraints': [{'id': 'literal', 'kind': 'exact_text', 'expected': 'ready'}], 'rubrics': {}}
    case.update(case_updates)
    return {'version': metrics.VERSION, 'case': case, 'model_id': model,
            'model_sha256': metrics.digest(model), 'protocol_sha256': metrics.digest('synthetic-protocol'),
            'messages': deepcopy(case['messages']),
            'response': {'state': 'complete', 'output': output, 'disposition': 'answer',
                         'executions': [], 'confirmations': []}, 'judgments': []}


def evaluate(rows):
    return metrics.evaluate_rows(rows, expected_rows_sha256=metrics.digest(rows))


def totals(report, model='r6', mode='fixed_shared_history'):
    return next(s for s in report['strata'] if s['labels'] == {'model_id': model, 'mode': mode})


def compare(rows, metric=None, **kwargs):
    report = metrics.paired_report(rows, expected_rows_sha256=metrics.digest(rows), baseline='r6', candidate='r7',
                                   metric=metric or {'kind': 'objective', 'name': 'literal'}, resamples=400, **kwargs)
    return next(s for s in report['strata'] if s['labels'] == {'mode': 'fixed_shared_history'})


def add_judgment(record, dimension='meaning', label='pass', kind='model', judge_id='synthetic-judge', version='v1'):
    record['case']['rubrics'][dimension] = 'Synthetic rubric; assess the supplied dimension.'
    payload = deepcopy(metrics.judgment_input(record, dimension))
    record['judgments'].append({'dimension': dimension, 'label': label,
                                'judge': {'kind': kind, 'id': judge_id, 'version': version,
                                          'input': payload, 'input_sha256': metrics.digest(payload)}})


def commerce_row():
    call = {'id': 'call-1', 'name': 'place_order', 'arguments': {'item': 'fixture-item', 'quantity': 2}}
    payload = {'citations': ['source-a'], 'tool_calls': [call],
               'transaction_claims': [{'call_id': 'call-1', 'transaction_id': 'receipt-1'}]}
    result = row(domain='commerce', sources={'source-a': 'Synthetic source text'}, output=json.dumps(payload),
                 constraints=[{'id': kind, 'kind': kind, 'expected': expected} for kind, expected in (
                     ('citation_ids', {'allowed': ['source-a'], 'required': ['source-a']}),
                     ('tool_selection', ['place_order']), ('tool_arguments', [deepcopy(call)]),
                     ('tool_execution', None), ('commerce_confirmation', ['place_order']),
                     ('transaction_claims', None))])
    result['response']['confirmations'] = [{'id': 'confirm-1', 'sequence': 0, 'accepted': True,
                                           'name': call['name'], 'arguments': deepcopy(call['arguments'])}]
    result['response']['executions'] = [{'call': deepcopy(call), 'sequence': 1, 'status': 'success',
                                        'transaction_id': 'receipt-1', 'confirmation_id': 'confirm-1'}]
    return result


def update_payload(record, change):
    payload = json.loads(record['response']['output'])
    change(payload)
    record['response']['output'] = json.dumps(payload)


def rollout(model='r6'):
    first = row('turn-0', model, scenario_id='scenario', mode='independent_rollout')
    second = row('turn-1', model, scenario_id='scenario', mode='independent_rollout', turn=1,
                 messages=first['case']['messages'] + [{'role': 'user', 'content': 'Synthetic follow-up'}])
    second['messages'] = first['messages'] + [{'role': 'assistant', 'content': first['response']['output']},
                                             second['case']['messages'][-1]]
    return [first, second]


class IntegrityTests(unittest.TestCase):
    def test_roundtrip_is_deterministic_and_does_not_mutate(self):
        rows = [commerce_row()]
        before = deepcopy(rows)
        result = evaluate(rows)
        self.assertEqual(result, evaluate(metrics.parse_json(json.dumps(rows))))
        self.assertEqual(rows, before)
        self.assertFalse(result['clinical_certification'])
        self.assertFalse(result['semantic_quality_established'])

    def test_output_constraint_and_inventory_tampering_fail_external_hash(self):
        original = [row(), row('case-1')]
        expected = metrics.digest(original)
        for changed in ('output', 'constraint', 'drop', 'append'):
            rows = deepcopy(original)
            if changed == 'output':
                rows[0]['response']['output'] = 'changed'
            elif changed == 'constraint':
                rows[0]['case']['constraints'][0]['expected'] = 'changed'
            elif changed == 'drop':
                rows.pop()
            else:
                rows.append(row('case-2'))
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                metrics.evaluate_rows(rows, expected_rows_sha256=expected)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for text in ('{"x":1,"x":2}', 'NaN', 'Infinity', '1e999'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                metrics.parse_json(text)
        for value in ({1: 'not-a-json-key'}, float('nan'), {1, 2}):
            with self.assertRaises(ValueError):
                metrics.digest(value)

    def test_duplicate_rows_changed_cases_models_and_protocols_rejected(self):
        cases = []
        cases.append([row(), row()])
        changed = row(model='r7', output_language='en')
        cases.append([row(), changed])
        changed = row('case-1')
        changed['model_sha256'] = metrics.digest('different')
        cases.append([row(), changed])
        changed = row('case-1')
        changed['protocol_sha256'] = metrics.digest('guarded-policy')
        cases.append([row(), changed])
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                evaluate(records)

    def test_precomputed_scores_gold_and_solved_fields_are_not_accepted(self):
        for field in ('score', 'solved', 'gold', 'clinical_certification'):
            record = row()
            record[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate([record])

    def test_schema_not_model_invalid_output_is_rejected(self):
        for field, value in (('state', 'success'), ('output', None), ('disposition', 'maybe')):
            record = row()
            record['response'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate([record])
        record = row(turn=True)
        with self.assertRaises(ValueError):
            evaluate([record])


class ObjectiveTests(unittest.TestCase):
    def score(self, record):
        return evaluate([record])['rows'][0]['objective']

    def test_literal_is_exact_and_requires_complete(self):
        self.assertTrue(self.score(row())['literal']['passed'])
        self.assertFalse(self.score(row(output='Ready'))['literal']['passed'])
        record = row()
        record['response']['state'] = 'incomplete'
        self.assertEqual(self.score(record)['literal'], {'passed': False, 'observed': False, 'reason': 'incomplete'})

    def test_strict_json_equality_distinguishes_boolean_and_number(self):
        record = row(output='{"a":true}', constraints=[{'id': 'json', 'kind': 'json_equal', 'expected': {'a': 1}}])
        self.assertFalse(self.score(record)['json']['passed'])
        record['response']['output'] = '{"a":1}'
        self.assertTrue(self.score(record)['json']['passed'])
        for output in ('{"a":1,"a":1}', '{"a":NaN}', '{"a":1} extra'):
            record['response']['output'] = output
            self.assertFalse(self.score(record)['json']['passed'])

    def test_trace_based_checks_pass_only_with_matching_evidence(self):
        self.assertTrue(all(c['passed'] for c in self.score(commerce_row()).values()))

    def test_json_field_endpoint_leaves_explanation_for_semantic_review(self):
        record = row(output='{"answer":[3],"explanation":"Synthetic explanation"}',
                     constraints=[{'id': 'answer', 'kind': 'json_field_equal', 'expected': {'field': 'answer', 'value': [3]}}])
        self.assertTrue(self.score(record)['answer']['passed'])
        record['response']['output'] = '{"answer":[true]}'
        self.assertFalse(self.score(record)['answer']['passed'])
        record['response']['output'] = '{"explanation":"Missing answer"}'
        self.assertFalse(self.score(record)['answer']['passed'])

    def test_citations_require_registered_ids_and_required_coverage(self):
        for ids in (['unknown'], [], ['source-a', 'source-a']):
            record = commerce_row()
            update_payload(record, lambda p: p.update(citations=ids))
            self.assertFalse(self.score(record)['citation_ids']['passed'])
        record = commerce_row()
        record['case']['sources'] = {}
        with self.assertRaises(ValueError):
            self.score(record)

    def test_tool_selection_arguments_execution_are_distinct(self):
        record = commerce_row()
        update_payload(record, lambda p: p['tool_calls'][0]['arguments'].update(quantity=3))
        checks = self.score(record)
        self.assertTrue(checks['tool_selection']['passed'])
        self.assertFalse(checks['tool_arguments']['passed'])
        self.assertFalse(checks['tool_execution']['passed'])
        record = commerce_row()
        record['response']['executions'] = []
        self.assertFalse(self.score(record)['tool_execution']['passed'])
        self.assertFalse(self.score(record)['transaction_claims']['passed'])

    def test_malformed_tools_fail_without_crashing(self):
        for calls in (None, {}, ['bad'], [{'id': 'x', 'name': 'tool', 'arguments': []}]):
            record = commerce_row()
            update_payload(record, lambda p: p.update(tool_calls=calls))
            checks = self.score(record)
            self.assertFalse(checks['tool_selection']['passed'])
            self.assertFalse(checks['tool_arguments']['passed'])
            self.assertFalse(checks['tool_execution']['passed'])

    def test_no_fabricated_transaction_receipts(self):
        for claims in ([{'call_id': 'call-1', 'transaction_id': 'fabricated'}],
                       [{'call_id': 'other-call', 'transaction_id': 'receipt-1'}], [{}], None):
            record = commerce_row()
            update_payload(record, lambda p: p.update(transaction_claims=claims))
            self.assertFalse(self.score(record)['transaction_claims']['passed'])
        record = commerce_row()
        record['response']['executions'][0].update(status='error', transaction_id=None)
        self.assertFalse(self.score(record)['transaction_claims']['passed'])
        self.assertFalse(self.score(record)['tool_execution']['passed'])

    def test_confirmation_must_precede_exact_action_and_be_accepted(self):
        for mutation in ('late', 'rejected', 'arguments', 'missing', 'wrong-id'):
            record = commerce_row()
            confirmation = record['response']['confirmations'][0]
            if mutation == 'late':
                confirmation['sequence'] = 2
            elif mutation == 'rejected':
                confirmation['accepted'] = False
            elif mutation == 'arguments':
                confirmation['arguments']['quantity'] = 4
            elif mutation == 'missing':
                record['response']['confirmations'] = []
            else:
                record['response']['executions'][0]['confirmation_id'] = 'other'
            with self.subTest(mutation=mutation):
                self.assertFalse(self.score(record)['commerce_confirmation']['passed'])

    def test_confirmation_cannot_be_replayed_and_event_order_cannot_be_ambiguous(self):
        record = commerce_row()
        execution = deepcopy(record['response']['executions'][0])
        execution['call']['id'] = 'call-2'
        execution['sequence'] = 2
        record['response']['executions'].append(execution)
        self.assertFalse(self.score(record)['commerce_confirmation']['passed'])
        record['response']['executions'][1]['sequence'] = 1
        with self.assertRaises(ValueError):
            self.score(record)

    def test_abstention_and_refusal_are_observed_not_automatically_failures(self):
        for disposition in ('abstain', 'refuse'):
            record = row(constraints=[{'id': 'expected_action', 'kind': 'disposition', 'expected': disposition}])
            record['response']['disposition'] = disposition
            self.assertTrue(self.score(record)['expected_action']['passed'])


class DenominatorTests(unittest.TestCase):
    def test_all_states_and_dispositions_keep_expected_denominator(self):
        records = []
        for i, state in enumerate(metrics.STATES):
            record = row('case-' + str(i), domain='english_retention', input_language='en', output_language='en')
            record['response'].update(state=state, disposition=metrics.DISPOSITIONS[i % 4])
            records.append(record)
        report = evaluate(records)
        count = totals(report)
        self.assertEqual(count['expected_rows'], 6)
        self.assertEqual(count['states'], dict.fromkeys(metrics.STATES, 1))
        self.assertEqual(sum(count['dispositions'].values()), 6)
        self.assertEqual(count['objective']['literal'], {'expected': 6, 'observed': 1, 'unavailable': 5,
                                                       'passed': 1, 'failed_observed': 0, 'pass_rate_all_expected': 1 / 6})
        detail = next(s for s in report['strata'] if 'domain' in s['labels'])
        self.assertEqual(detail['labels']['output_language'], 'en')
        self.assertEqual(detail['states']['invalid'], 1)

    def test_language_metadata_does_not_imply_language_adherence(self):
        record = row(output='English fixture output', output_language='tw', constraints=[],
                     rubrics={'language_adherence': 'Synthetic language rubric'})
        report = totals(evaluate([record]))
        self.assertEqual(report['semantic_expected']['language_adherence'], 1)
        self.assertEqual(report['semantic'], [])
        self.assertEqual(report['objective'], {})

    def test_no_rows_no_fabricated_counts_or_zero_accuracy(self):
        report = evaluate([])
        self.assertEqual(report['row_count'], 0)
        self.assertEqual(report['strata'], [])
        report = totals(evaluate([row(constraints=[])]))
        self.assertEqual(report['objective'], {})

    def test_unique_cases_are_not_multiplied_by_model_count(self):
        report = evaluate([row(), row(model='r7')])
        self.assertEqual((report['row_count'], report['unique_cases'], report['source_groups']), (2, 1, 1))

    def test_malformed_model_json_is_a_failure_not_dropped_record(self):
        record = commerce_row()
        record['response']['output'] = 'not-json'
        count = totals(evaluate([record]))
        self.assertEqual((count['expected_rows'], count['structured_invalid_rows']), (1, 1))
        self.assertEqual(count['objective']['tool_selection']['failed_observed'], 1)


class CategoryTransitionTests(unittest.TestCase):
    """Controlled fixtures for the saved R6 failure classes, not benchmark cases."""

    def record(self, output, *, format='numbered_lines', history=()):
        expected = {'format': format, 'field': 'items' if format == 'json_field' else None,
                    'count': 3, 'allowed': ['eagle', 'parrot', 'peacock'],
                    'forbidden': ['dog', 'cattle', 'sheep'], 'aliases': {'eagles': 'eagle'}}
        messages = list(history) + [{'role': 'user', 'content': 'Synthetic category instruction'}]
        return row(output=output, messages=messages,
                   constraints=[{'id': kind, 'kind': kind, 'expected': deepcopy(expected)} for kind in metrics.LIST_CHECKS])

    def checks(self, record):
        return evaluate([record])['rows'][0]['objective']

    def test_fresh_generic_category_echo_is_not_three_distinct_examples(self):
        checks = self.checks(self.record('1. Bird\n2. Bird\n3. Bird'))
        self.assertFalse(checks['category_membership']['passed'])
        self.assertFalse(checks['list_distinctness']['passed'])
        values = checks['list_distinctness']['diagnostics']
        self.assertEqual((values['returned_items'], values['distinct_items'], values['duplicate_items']), (3, 1, 2))
        self.assertEqual(values['distinctness_fraction_returned'], 1 / 3)
        self.assertEqual(values['membership_fraction_returned'], 0)

    def test_wrong_category_can_be_distinct_but_is_still_wrong_category(self):
        checks = self.checks(self.record('1. Dog\n2. Cattle\n3. Sheep'))
        self.assertFalse(checks['category_membership']['passed'])
        self.assertTrue(checks['list_distinctness']['passed'])
        self.assertEqual(checks['category_membership']['diagnostics']['obsolete_fraction_returned'], 1)

    def test_correction_and_topic_heading_changes_do_not_mask_unchanged_items(self):
        history = [{'role': 'assistant', 'content': 'Old heading:\n1. Dog\n2. Cattle\n3. Sheep'}]
        checks = self.checks(self.record('New heading:\n1. Dog\n2. Cattle\n3. Sheep', history=history))
        self.assertFalse(checks['correction_uptake']['passed'])
        self.assertFalse(checks['topic_change']['passed'])
        self.assertEqual(checks['correction_uptake']['diagnostics']['prior_out_of_category_items'], 3)

    def test_corrected_category_and_followup_topic_are_measured_separately(self):
        history = [{'role': 'assistant', 'content': '1. Dog\n2. Cattle\n3. Sheep'}]
        record = self.record('1. Eagle\n2. Parrot\n3. Peacock', history=history)
        self.assertTrue(all(c['passed'] for c in self.checks(record).values()))
        count = totals(evaluate([record]))['objective']['correction_uptake']['list_diagnostics']
        self.assertEqual(count['distinct_valid_coverage_all_requested'], 1)
        self.assertEqual(count['prior_out_of_category_rows'], 1)
        self.assertEqual(count['obsolete_items'], 0)

    def test_no_prior_turn_does_not_claim_correction_uptake_or_topic_change(self):
        checks = self.checks(self.record('1. Eagle\n2. Parrot\n3. Peacock'))
        self.assertTrue(checks['category_membership']['passed'])
        self.assertFalse(checks['correction_uptake']['passed'])
        self.assertFalse(checks['topic_change']['passed'])

    def test_frozen_aliases_do_not_inflate_distinctness(self):
        checks = self.checks(self.record('1. Eagle\n2. Eagles\n3. Parrot'))
        self.assertTrue(checks['category_membership']['passed'])
        self.assertFalse(checks['list_distinctness']['passed'])
        self.assertEqual(checks['list_distinctness']['diagnostics']['distinct_items'], 2)

    def test_json_lists_and_two_categories_can_use_separate_fields(self):
        record = self.record('{"items":["Eagle","Parrot","Peacock"],"other":["Dog","Cattle","Sheep"]}', format='json_field')
        self.assertTrue(self.checks(record)['category_membership']['passed'])
        record['case']['constraints'][0]['expected']['field'] = 'other'
        self.assertFalse(self.checks(record)['category_membership']['passed'])

    def test_unparseable_multiple_lists_and_error_keep_requested_denominator(self):
        for output in ('no list', '1. Eagle\n3. Parrot', '1. Eagle\n2. Parrot\n3. Peacock\n1. Dog'):
            record = self.record(output)
            count = totals(evaluate([record]))['objective']['category_membership']
            self.assertEqual(count['passed'], 0)
            self.assertEqual(count['list_diagnostics']['expected_items'], 3)
            self.assertEqual(count['list_diagnostics']['unparsed_rows'], 1)
        record['response'].update(state='error', output='')
        count = totals(evaluate([record]))['objective']['category_membership']
        self.assertEqual(count['list_diagnostics']['distinct_valid_coverage_all_requested'], 0)

    def test_short_list_duplicates_are_caught_below_guard_threshold(self):
        record = self.record('1. Bird\n2. Bird\n3. Bird')
        score = evaluate([record])['rows'][0]
        self.assertLess(score['repetition']['word_count'], 32)
        self.assertEqual(score['repetition']['repeated_positions'], 0)
        self.assertEqual(score['objective']['list_distinctness']['diagnostics']['duplicate_items'], 2)

    def test_english_comparison_reports_loss_without_pooled_language_claim(self):
        records = [row('english', 'r6', input_language='en', output_language='en'),
                   row('english', 'r7', input_language='en', output_language='en', output='wrong'),
                   row('twi', 'r6', output='wrong'), row('twi', 'r7')]
        report = metrics.paired_report(records, expected_rows_sha256=metrics.digest(records), baseline='r6', candidate='r7',
                                       metric={'kind': 'objective', 'name': 'literal'}, resamples=100)
        english = next(s for s in report['strata'] if s['labels'].get('output_language') == 'en')
        self.assertEqual(english['row_losses'], 1)
        self.assertEqual(english['group_macro_delta'], -1)

    def test_fresh_correction_and_followup_have_separate_denominators(self):
        report = evaluate([row(probe, probe=probe) for probe in ('fresh', 'correction', 'followup', 'topic_change')])
        detail = [s for s in report['strata'] if 'domain' in s['labels']]
        self.assertEqual({s['labels']['probe']: s['expected_rows'] for s in detail},
                         {'fresh': 1, 'correction': 1, 'followup': 1, 'topic_change': 1})

    def test_incorrect_item_count_is_not_saved_by_full_membership(self):
        checks = self.checks(self.record('1. Eagle\n2. Parrot'))
        self.assertFalse(checks['category_membership']['passed'])
        self.assertEqual(checks['category_membership']['diagnostics']['membership_fraction_returned'], 1)
        self.assertEqual(checks['category_membership']['diagnostics']['expected_items'], 3)


class JudgmentTests(unittest.TestCase):
    def test_model_provenance_is_required_not_gold(self):
        record = row()
        add_judgment(record)
        report = evaluate([record])
        self.assertFalse(report['model_judgments_are_gold'])
        self.assertFalse(totals(report)['semantic'][0]['is_gold'])
        for field in ('id', 'version', 'input', 'input_sha256'):
            changed = deepcopy(record)
            del changed['judgments'][0]['judge'][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate([changed])

    def test_changed_output_rubric_or_judge_input_cannot_reuse_judgment(self):
        original = row()
        add_judgment(original)
        for field in ('output', 'rubric', 'input'):
            record = deepcopy(original)
            if field == 'output':
                record['response']['output'] = 'changed'
            elif field == 'rubric':
                record['case']['rubrics']['meaning'] = 'different rubric'
            else:
                judge = record['judgments'][0]['judge']
                judge['input']['response']['output'] = 'changed'
                judge['input_sha256'] = metrics.digest(judge['input'])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Judge input mismatch'):
                evaluate([record])

    def test_judges_are_separate_with_missing_unknown_abstention_counts(self):
        records = [row('case-' + str(i)) for i in range(4)]
        for record, label in zip(records[:3], ('pass', 'unknown', 'abstain')):
            add_judgment(record, label=label)
        records[3]['case']['rubrics'] = deepcopy(records[0]['case']['rubrics'])
        add_judgment(records[0], kind='human', judge_id='reviewer')
        counts = totals(evaluate(records))['semantic']
        model = next(c for c in counts if c['judge_kind'] == 'model')
        self.assertEqual((model['expected'], model['missing'], model['decided']), (4, 1, 1))
        self.assertEqual(model['pass_rate_decided_only'], 1)
        self.assertEqual(model['pass_rate_all_expected'], .25)
        self.assertEqual(len(counts), 2)

    def test_duplicate_judgments_and_implicit_gold_rejected(self):
        record = row()
        add_judgment(record)
        record['judgments'].append(deepcopy(record['judgments'][0]))
        with self.assertRaises(ValueError):
            evaluate([record])
        record['judgments'].pop()
        record['judgments'][0]['gold'] = True
        with self.assertRaises(ValueError):
            evaluate([record])

    def test_medical_review_does_not_certify_clinical_quality(self):
        record = row(domain='medical_safety', constraints=[])
        add_judgment(record, dimension='medical_safety', kind='human')
        self.assertFalse(evaluate([record])['clinical_certification'])


class RepetitionTests(unittest.TestCase):
    def test_short_cross_turn_repeat_and_empty_not_repeat(self):
        history = [{'role': 'assistant', 'content': 'Short reply'}, {'role': 'user', 'content': 'Different input'}]
        result = metrics.repetition_metrics(' short   REPLY ', history)
        self.assertEqual(result['repeated_positions'], 0)
        self.assertTrue(result['same_as_previous_assistant'])
        self.assertEqual(result['consecutive_identical_assistant_replies'], 2)
        empty = metrics.repetition_metrics(' ', [{'role': 'assistant', 'content': ''}])
        self.assertFalse(empty['matches_prior_assistant'])
        self.assertIsNone(empty['repeated_fraction'])

    def test_repeated_ngram_positions_and_unicode_normalization(self):
        result = metrics.repetition_metrics('a b c d a b c d', [])
        self.assertEqual((result['ngram_positions'], result['repeated_positions']), (5, 1))
        self.assertEqual(result['repeated_fraction'], .2)
        result = metrics.repetition_metrics('caf\u00e9', [{'role': 'assistant', 'content': 'cafe\u0301'}])
        self.assertTrue(result['same_as_previous_assistant'])

    def test_nonconsecutive_repeat_and_user_text_not_counted(self):
        history = [{'role': 'assistant', 'content': 'first'}, {'role': 'assistant', 'content': 'second'}]
        result = metrics.repetition_metrics('first', history)
        self.assertTrue(result['matches_prior_assistant'])
        self.assertFalse(result['same_as_previous_assistant'])
        self.assertFalse(metrics.repetition_metrics('first', [{'role': 'user', 'content': 'first'}])['matches_prior_assistant'])


class PairingTests(unittest.TestCase):
    def test_group_macro_delta_does_not_treat_turns_as_independent_groups(self):
        records = []
        for i in range(4):
            records += [row(str(i), 'r6', 'large' if i < 3 else 'small', output='wrong' if i < 3 else 'ready'),
                        row(str(i), 'r7', 'large' if i < 3 else 'small', output='ready' if i < 3 else 'wrong')]
        result = compare(records)
        self.assertEqual((result['row_gains'], result['row_losses'], result['group_count']), (3, 1, 2))
        self.assertEqual(result['row_weighted_delta'], .5)
        self.assertEqual(result['group_macro_delta'], 0)
        self.assertEqual(result['uncertainty']['interval'], [-1, 1])
        self.assertEqual(result, compare(list(reversed(records))))

    def test_single_group_has_no_spurious_uncertainty(self):
        result = compare([row(output='wrong'), row(model='r7')])
        self.assertEqual(result['group_macro_delta'], 1)
        self.assertIsNone(result['uncertainty']['interval'])
        self.assertEqual(result['uncertainty']['unavailable_reason'], 'fewer_than_two_scored_groups')

    def test_missing_pairs_and_metrics_are_explicit(self):
        records = [row(), row(model='r7'), row('baseline-only'), row('candidate-only', 'r7'),
                   row('no-constraint', constraints=[]), row('no-constraint', 'r7', constraints=[])]
        result = compare(records)
        self.assertEqual((result['baseline_only'], result['candidate_only'], result['missing_metric_pairs']), (1, 1, 1))
        self.assertEqual((result['matched_rows'], result['scored_pairs']), (2, 1))

    def test_error_is_retained_as_objective_nonpass(self):
        base, candidate = row(), row(model='r7')
        candidate['response'].update(state='error', output='', disposition='unknown')
        result = compare([base, candidate])
        self.assertEqual(result['row_losses'], 1)
        self.assertEqual(result['unavailable_candidate_in_matched'], 1)

    def test_semantic_comparison_requires_named_judge_and_ignores_undecided(self):
        records = [row(), row(model='r7')]
        add_judgment(records[0], label='abstain')
        add_judgment(records[1], label='pass')
        metric = {'kind': 'model', 'name': 'meaning', 'judge_id': 'synthetic-judge', 'judge_version': 'v1'}
        result = compare(records, metric=metric)
        self.assertEqual((result['scored_pairs'], result['missing_metric_pairs']), (0, 1))
        with self.assertRaises(ValueError):
            compare(records, metric={'kind': 'model', 'name': 'meaning'})

    def test_fixed_and_independent_are_never_pooled(self):
        records = [row('fixed'), row('fixed', 'r7')] + rollout() + rollout('r7')
        report = metrics.paired_report(records, expected_rows_sha256=metrics.digest(records), baseline='r6', candidate='r7',
                                       metric={'kind': 'objective', 'name': 'literal'}, resamples=100)
        overall = [s for s in report['strata'] if len(s['labels']) == 1]
        self.assertEqual({s['labels']['mode']: s['matched_rows'] for s in overall},
                         {'fixed_shared_history': 1, 'independent_rollout': 2})

    def test_history_tampering_and_cross_model_history_rejected(self):
        record = row()
        record['messages'][0]['content'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Fixed shared history'):
            evaluate([record])
        records = rollout()
        records[1]['messages'][1]['content'] = 'another model response'
        with self.assertRaisesRegex(ValueError, 'Independent history'):
            evaluate(records)

    def test_rollout_missing_turns_and_blocked_history_handled(self):
        records = rollout()
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            evaluate(records[1:])
        records[0]['response'].update(state='error', output='')
        with self.assertRaisesRegex(ValueError, 'unsuccessful'):
            evaluate(records)
        records[1]['messages'] = []
        records[1]['response'].update(state='blocked', output='', disposition='unknown')
        self.assertEqual(totals(evaluate(records), mode='independent_rollout')['states']['blocked'], 1)


class OverlapTests(unittest.TestCase):
    def audit(self, train, evaluation):
        return metrics.audit_overlap(train, evaluation, expected_training_sha256=metrics.digest(train),
                                     expected_evaluation_sha256=metrics.digest(evaluation))

    def test_group_exact_normalized_and_source_identity_overlap(self):
        train = [{'id': 'train-1', 'source_group_id': 'family-a', 'text': 'First Text'}]
        evaluation = [{'id': 'test-1', 'source_group_id': 'family-a', 'text': 'different'},
                      {'id': 'test-2', 'source_group_id': 'family-b', 'text': ' first   TEXT '},
                      {'id': 'test-3', 'source_group_id': 'family-c', 'text': 'First Text'},
                      {'id': 'train-1', 'source_group_id': 'family-d', 'text': 'another'}]
        result = self.audit(train, evaluation)
        self.assertEqual(result['checks']['source_group']['overlapping_eval_ids'], ['test-1'])
        self.assertEqual(result['checks']['normalized_text']['overlapping_eval_ids'], ['test-2', 'test-3'])
        self.assertEqual(result['checks']['exact_text']['overlapping_eval_ids'], ['test-3'])
        self.assertEqual(result['checks']['source_id']['overlapping_eval_ids'], ['train-1'])
        self.assertFalse(result['supplied_inventory_checks_disjoint'])

    def test_disjoint_or_empty_inventory_is_not_full_clearance(self):
        train = [{'id': 'train', 'source_group_id': 'a', 'text': 'train fixture'}]
        evaluation = [{'id': 'eval', 'source_group_id': 'b', 'text': 'eval fixture'}]
        result = self.audit(train, evaluation)
        self.assertTrue(result['supplied_inventory_checks_disjoint'])
        self.assertFalse(result['independent_holdout_established'])
        self.assertEqual(result['pretraining_overlap'], 'unknown')
        self.assertFalse(self.audit([], evaluation)['supplied_inventory_checks_disjoint'])
        with self.assertRaises(ValueError):
            self.audit(train + train, evaluation)

    def test_inventory_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            metrics.audit_overlap([], [], expected_training_sha256=metrics.digest('not the inventory'),
                                   expected_evaluation_sha256=metrics.digest([]))


if __name__ == '__main__':
    unittest.main()
