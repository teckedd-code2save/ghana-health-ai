#!/usr/bin/env python3
"""Freeze source-protected, AI-authored R7 development probes without inference."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import TypedDict

import r7_eval_metrics as metrics
from prepare_understanding_run import HoldoutIndex, normalized as overlap_normalized

VERSION = 'r7-behavior-eval-v1'
LANGUAGES = ('tw', 'en')
REGRESSION_SESSIONS = ('18d13da7638848258cd345c3b5f163d2', 'd9bd93ed80fe4fb0a50f3dbbaa418aab')
require = metrics.require


class RolloutScenario(TypedDict):
    id: str
    source_group_id: str
    language: str
    system_message: str
    user_turns: list[str]
    expected_per_turn: list[metrics.EvalCase]


def bilingual(en, tw):
    return {'en': en, 'tw': tw}


def source_scenarios() -> list[dict]:
    """Distinct authored worlds; languages/history probes are views, never new groups."""
    return [
        {'id': 'categories-01', 'family': 'category_membership',
         'facts': bilingual('In this exercise, group S contains triangle, square, circle. Group C contains jar, cup, bowl.',
                            'W\u0254 saa adesua yi mu, kuw S ne triangle, square, circle. Kuw C ne jar, cup, bowl.'),
         'queries': [bilingual('List the three members of group S.', 'B\u0254 nne\u025bma abi\u025bsa a \u025bw\u0254 kuw S mu no din.'),
                     bilingual('Now list the three members of group C.', 'Afei b\u0254 nne\u025bma abi\u025bsa a \u025bw\u0254 kuw C mu no din.'),
                     bilingual('Change task: how many members are there in both groups altogether?',
                               'Sesa adwuma no: nne\u025bma a \u025bw\u0254 akuw abien no mu nyinaa dodow y\u025b ahe?')],
         'answers': [['triangle', 'square', 'circle'], ['jar', 'cup', 'bowl'], [6]],
         'wrong': ['jar', 'cup', 'bowl']},
        {'id': 'reference-01', 'family': 'reference_resolution',
         'facts': bilingual('The green disk is in drawer M. The yellow disk is in drawer N. In note X, "it" refers to the yellow disk.',
                            'Disk ahabammono no w\u0254 drawer M mu. Disk akok\u0254srade no w\u0254 drawer N mu. W\u0254 krataa X mu, "\u025bno" gyina h\u0254 ma disk akok\u0254srade no.'),
         'queries': [bilingual('In note X, which drawer contains it? Return the drawer letter.',
                               'W\u0254 krataa X mu, drawer b\u025bn na \u025bno w\u0254 mu? Fa drawer no nkyer\u025bwde ma.'),
                     bilingual('Which drawer contains the other disk, the green one?',
                               'Drawer b\u025bn na disk foforo no, nea \u025by\u025b ahabammono no, w\u0254 mu?'),
                     bilingual('Change task: how many disks are mentioned in the source?',
                               'Sesa adwuma no: disk dodow ahe na w\u0254aka ho as\u025bm w\u0254 fibea no mu?')],
         'answers': [['N'], ['M'], [2]], 'wrong': ['M']},
        {'id': 'entity-01', 'family': 'entity_binding',
         'facts': bilingual('Practice parcel T has mark K7 and destination D4. Practice parcel U has mark K2 and destination D8. These are fictional labels, not real orders.',
                            'Adesua ade T agyiraehy\u025bde ne K7, na \u025brek\u0254 D4. Adesua ade U agyiraehy\u025bde ne K2, na \u025brek\u0254 D8. Eyinom y\u025b adesua din, \u025bny\u025b nne\u025bma a obi at\u0254 ankasa.'),
         'queries': [bilingual('What is the mark on parcel T?', 'D\u025bn ne ade T agyiraehy\u025bde?'),
                     bilingual('What is the mark on parcel U instead?', 'Na ade U nso agyiraehy\u025bde ne d\u025bn?'),
                     bilingual('Change task: give the destination of parcel T, not its mark.',
                               'Sesa adwuma no: ka baabi a ade T rek\u0254, \u025bny\u025b n\u2019agyiraehy\u025bde.')],
         'answers': [['K7'], ['K2'], ['D4']], 'wrong': ['K2']},
        {'id': 'time-01', 'family': 'time_ordering',
         'facts': bilingual('In a fictional same-day schedule, event X starts at 09:10, event Y at 10:40, and event Z at 08:55. All times use the same time zone.',
                            'W\u0254 adesua nhyehy\u025be bi mu, da koro no ara, dwumadi X fi ase 09:10, Y fi ase 10:40, na Z fi ase 08:55. Bere no nyinaa w\u0254 bere beae koro mu.'),
         'queries': [bilingual('Which event starts first? Return its letter.', 'Dwumadi b\u025bn na efi ase kan? Fa ne nkyer\u025bwde ma.'),
                     bilingual('Which event starts after X?', 'Dwumadi b\u025bn na efi ase w\u0254 X akyi?'),
                     bilingual('Change task: give the start time of X, not an event letter.',
                               'Sesa adwuma no: ka bere a X fi ase, \u025bny\u025b dwumadi no nkyer\u025bwde.')],
         'answers': [['Z'], ['Y'], ['09:10']], 'wrong': ['Y']},
        {'id': 'quantity-01', 'family': 'quantity_update',
         'facts': bilingual('A practice shelf starts with 14 counters. Five are removed, then four are added. Separately, there are three trays with two counters on each tray.',
                            'Adesua shelf bi w\u0254 nne\u025bma 14. W\u0254yi 5 fi mu, na afei w\u0254de 4 ka ho. W\u0254 beae foforo no, tray 3 w\u0254 h\u0254, na tray biara so w\u0254 nne\u025bma 2.'),
         'queries': [bilingual('How many counters remain immediately after removing five, before adding four?',
                               'Nne\u025bma ahe na \u025bka bere a w\u0254ayi 5 no, ansa na w\u0254de 4 b\u025bka ho?'),
                     bilingual('How many are on that shelf after adding the four as well?',
                               'Bere a w\u0254de 4 no nso aka ho no, nne\u025bma ahe na \u025bw\u0254 shelf no so?'),
                     bilingual('Change task: how many counters are on the three separate trays altogether?',
                               'Sesa adwuma no: nne\u025bma ahe na \u025bw\u0254 tray 3 a \u025bw\u0254 beae foforo no so nyinaa?')],
         'answers': [[9], [13], [6]], 'wrong': [19]},
        {'id': 'negation-01', 'family': 'negation_exclusion',
         'facts': bilingual('In this exercise, option A is permitted, option B is not permitted, and option C is permitted. A temporary instruction excludes C from the first selection.',
                            'W\u0254 saa adesua yi mu, w\u0254ama A ho kwan, w\u0254mma B ho kwan, na w\u0254ama C ho kwan. Akwankyer\u025b bi ka s\u025b mma C nka nea wob\u025bpa kan no ho.'),
         'queries': [bilingual('Which permitted option remains when C is excluded?', 'S\u025b wuyi C fi mu a, nea w\u0254ama ho kwan no bi b\u025bn na \u025bka?'),
                     bilingual('Which option is not permitted according to the source?', 'S\u025bnea fibea no ka no, nea w\u0254mma ho kwan no ne d\u025bn?'),
                     bilingual('Change task: ignore the temporary exclusion and list all permitted options in source order.',
                               'Sesa adwuma no: yi bere tiaa no mu akwankyer\u025b no to nky\u025bn, na kyer\u025bw nea w\u0254ama ho kwan nyinaa s\u025bnea \u025btoatoa so w\u0254 fibea no mu.')],
         'answers': [['A'], ['B'], ['A', 'C']], 'wrong': ['C']},
        {'id': 'spatial-01', 'family': 'spatial_relations',
         'facts': bilingual('On a fictional map, point R is west of S. Point T is north of S. Point U is south of S. Use the literal direction labels west, east, north, south.',
                            'W\u0254 adesua asase mfonini bi so, R w\u0254 S at\u0254e (west). T w\u0254 S atifi (north). U w\u0254 S anafo (south). "east" kyer\u025b apuei.'),
         'queries': [bilingual('Which point is north of S?', 'Beae b\u025bn na \u025bw\u0254 S atifi (north)?'),
                     bilingual('Which point is east of R?', 'Beae b\u025bn na \u025bw\u0254 R apuei (east)?'),
                     bilingual('Change task: which point is south of S?', 'Sesa adwuma no: beae b\u025bn na \u025bw\u0254 S anafo (south)?')],
         'answers': [['T'], ['S'], ['U']], 'wrong': ['U']},
        {'id': 'state-01', 'family': 'state_revision',
         'facts': bilingual('The first record says token Q has label red. A later update replaces Q\'s label with blue. Token P keeps label green. The later update is authoritative.',
                            'Krataa a edi kan no ka s\u025b token Q din ne red. Ns\u025bm foforo a \u025bbaa akyiri no sesa Q din no y\u025b\u025b blue. Token P din da so ara y\u025b green. Ns\u025bm a \u025bbaa akyiri no na di so.'),
         'queries': [bilingual('What is Q\'s current label?', 'D\u025bn ne Q din seesei?'),
                     bilingual('What was Q\'s label before the update?', 'D\u025bn na na \u025by\u025b Q din ansa na ns\u025bm foforo no reba?'),
                     bilingual('Change task: what is P\'s unchanged label?', 'Sesa adwuma no: d\u025bn ne P din a \u025bnsesae no?')],
         'answers': [['blue'], ['red'], ['green']], 'wrong': ['red']},
    ]


def validate_sources(sources: list[dict]) -> None:
    ids, contents = set(), set()
    require(len({source['family'] for source in sources}) >= 8, 'At least eight semantic families required')
    for source in sources:
        metrics.fields(source, 'id family facts queries answers wrong')
        require(metrics.nonempty(source['id']) and source['id'] not in ids, 'Duplicate source scenario')
        ids.add(source['id'])
        require(set(source['facts']) == set(LANGUAGES) and len(source['queries']) == len(source['answers']) == 3,
                'Matched languages and three scripted steps required')
        require(all(set(q) == set(LANGUAGES) and all(metrics.nonempty(v) for v in q.values())
                    for q in source['queries']) and all(metrics.nonempty(v) for v in source['facts'].values()),
                'Missing matched question/facts')
        require(all(type(answer) is list and answer for answer in source['answers'])
                and type(source['wrong']) is list and source['wrong']
                and metrics.digest(source['wrong']) != metrics.digest(source['answers'][0]), 'Invalid expected/wrong targets')
        fingerprint = metrics.digest({'facts': overlap_normalized(source['facts']['en']),
                                      'queries': [overlap_normalized(q['en']) for q in source['queries']],
                                      'answers': source['answers']})
        require(fingerprint not in contents, 'Same semantic scenario under another ID')
        contents.add(fingerprint)


def system_message(language: str) -> str:
    return bilingual(
        'Use only the supplied exercise facts. Return one JSON object with answer (an array of exact IDs, labels, or numbers), citations (an array of source IDs), and explanation (one English sentence). Keep multiple answers in source order. Do not invent facts or perform real actions.',
        'Fa ns\u025bm a \u025bw\u0254 adesua fibea no mu nko ara di dwuma. Fa JSON object baako ma: answer (array a din, ID anaa akontaahy\u025bde w\u0254 mu), citations (array a fibea ID w\u0254 mu), ne explanation (Twi kasamu baako). S\u025b mmuae no d\u0254\u0254so a, fa fibea no mu ntoatoaso di dwuma. Mfa ns\u025bm foforo mm\u0254 ho, na ny\u025b ade ankasa biara.')[language]


def answer_text(answer: list, source_id: str, language: str) -> str:
    return json.dumps({'answer': answer, 'citations': [source_id],
                       'explanation': bilingual('This follows the supplied exercise facts.',
                                                'Eyi di adesua no mu ns\u025bm no so.')[language]}, ensure_ascii=False)


def constraints(source: dict, step: int, probe: str, source_id: str) -> list[metrics.Constraint]:
    answer = deepcopy(source['answers'][step])
    checks = [{'id': 'answer', 'kind': 'json_field_equal', 'expected': {'field': 'answer', 'value': answer}},
              {'id': 'citations', 'kind': 'citation_ids', 'expected': {'allowed': [source_id], 'required': [source_id]}}]
    if source['family'] == 'category_membership' and all(type(value) is str for value in answer):
        expected = {'format': 'json_field', 'field': 'answer', 'count': len(answer),
                    'allowed': answer, 'forbidden': source['answers'][1 - step], 'aliases': {}}
        for kind in ('category_membership', 'list_distinctness') + (('correction_uptake',) if probe == 'correction' else ()):
            checks.append({'id': kind, 'kind': kind, 'expected': deepcopy(expected)})
    return checks


def make_case(source: dict, language: str, mode: str, probe: str, turn: int,
              messages: list[metrics.Message], step: int) -> metrics.EvalCase:
    group = VERSION + ':' + source['id']
    scenario_id = group + ':' + language + ':' + mode
    source_id = 'source-' + source['id']
    rubrics = {'meaning': 'Assess requested entities, relations, negation, quantities and intent against supplied facts.',
               'source_grounding': 'Assess whether the explanation is supported by the supplied source, not just cited IDs.',
               'citation_support': 'Assess whether cited facts support the actual answer and explanation.',
               'language_adherence': 'Assess explanation language and naturalness independently of symbolic answer correctness.'}
    if language == 'en':
        rubrics['english_retention'] = 'Assess useful, grammatical English and preserved task meaning, not Twi performance.'
    if probe != 'fresh':
        rubrics['multiturn_usefulness'] = 'Assess use of relevant context and a useful response to the current turn.'
    if probe in ('correction', 'topic_change'):
        rubrics['correction_uptake' if probe == 'correction' else 'topic_change'] = (
            'Assess whether the current request is followed instead of repeating the previous answer or obsolete topic.')
    return {'id': scenario_id + ':' + probe, 'scenario_id': scenario_id, 'source_group_id': group,
            'domain': source['family'], 'input_language': language, 'output_language': language,
            'mode': mode, 'turn': turn, 'probe': probe, 'messages': deepcopy(messages),
            'sources': {source_id: source['facts'][language]},
            'constraints': constraints(source, step, probe, source_id), 'rubrics': rubrics}


def build_views(sources: list[dict]) -> tuple[list[metrics.EvalCase], list[RolloutScenario]]:
    validate_sources(sources)
    fixed, rollouts = [], []
    for source in sources:
        for language in LANGUAGES:
            source_id = 'source-' + source['id']
            first = source_id + ':\n' + source['facts'][language] + '\n\n' + source['queries'][0][language]
            system = {'role': 'system', 'content': system_message(language)}
            initial = [system, {'role': 'user', 'content': first}]
            fixed.append(make_case(source, language, 'fixed_shared_history', 'fresh', 0, initial, 0))
            correction = bilingual('The previous answer is wrong. Recheck the original source, not the previous answer. ',
                                   'Mmuae a edi kan no ny\u025b nokware. San hw\u025b fibea no mu ns\u025bm, \u025bny\u025b mmuae a edi kan no. ')[language]
            correction += source['queries'][0][language]
            corrected = initial + [{'role': 'assistant', 'content': answer_text(source['wrong'], source_id, language)},
                                    {'role': 'user', 'content': correction}]
            fixed.append(make_case(source, language, 'fixed_shared_history', 'correction', 1, corrected, 0))
            history = initial
            for step, probe in ((1, 'followup'), (2, 'topic_change')):
                history = history + [{'role': 'assistant', 'content': answer_text(source['answers'][step - 1], source_id, language)},
                                     {'role': 'user', 'content': source['queries'][step][language]}]
                fixed.append(make_case(source, language, 'fixed_shared_history', probe, step, history, step))
            user_turns = [first] + [source['queries'][step][language] for step in (1, 2)]
            cases = [make_case(source, language, 'independent_rollout', probe, step,
                               [system] + [{'role': 'user', 'content': text} for text in user_turns[:step + 1]], step)
                     for step, probe in enumerate(('fresh', 'followup', 'topic_change'))]
            rollouts.append({'id': cases[0]['scenario_id'], 'source_group_id': cases[0]['source_group_id'],
                             'language': language, 'system_message': system['content'],
                             'user_turns': user_turns, 'expected_per_turn': cases})
    validate_views(fixed, rollouts)
    return fixed, rollouts


def all_cases(fixed: list[metrics.EvalCase], rollouts: list[RolloutScenario]) -> list[metrics.EvalCase]:
    return fixed + [case for scenario in rollouts for case in scenario['expected_per_turn']]


def validate_views(fixed, rollouts) -> None:
    ids = set()
    for scenario in rollouts:
        metrics.fields(scenario, 'id source_group_id language system_message user_turns expected_per_turn')
        require(scenario['id'] not in ids and len(scenario['user_turns']) == len(scenario['expected_per_turn']),
                'Duplicate/misaligned rollout scenario')
        ids.add(scenario['id'])
        for turn, case in enumerate(scenario['expected_per_turn']):
            expected_messages = [{'role': 'system', 'content': scenario['system_message']}]
            expected_messages += [{'role': 'user', 'content': text} for text in scenario['user_turns'][:turn + 1]]
            require(case['scenario_id'] == scenario['id'] and case['source_group_id'] == scenario['source_group_id']
                    and case['input_language'] == case['output_language'] == scenario['language']
                    and case['turn'] == turn and case['mode'] == 'independent_rollout'
                    and case['messages'] == expected_messages, 'Rollout case/script mismatch')
    require(all(case['mode'] == 'fixed_shared_history' for case in fixed), 'Fixed cases have wrong mode')
    cases = all_cases(fixed, rollouts)
    # These temporary placeholders validate inputs only; no model-result file is emitted.
    rows = [{'version': metrics.VERSION, 'case': case, 'model_id': 'unexecuted-schema-check',
             'model_sha256': metrics.digest('unexecuted'), 'protocol_sha256': metrics.digest(VERSION),
             'messages': case['messages'] if case['mode'] == 'fixed_shared_history' or case['turn'] == 0 else [],
             'response': {'state': 'pending', 'output': '', 'disposition': 'unknown', 'executions': [], 'confirmations': []},
             'judgments': []} for case in cases]
    metrics.validate_rows(rows, expected_rows_sha256=metrics.digest(rows))


def remote_fixed(case: metrics.EvalCase) -> dict:
    return {'id': case['id'], 'messages': deepcopy(case['messages'])}


def remote_rollout(scenario: RolloutScenario) -> dict:
    return {key: deepcopy(scenario[key]) for key in ('id', 'system_message', 'user_turns')}


def regression_cases(database: Path) -> tuple[list[metrics.EvalCase], dict]:
    """Read only the two specifically authorized saved sessions, without copying the DB."""
    require(database.is_file(), 'Saved regression database missing')
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        records = db.execute('SELECT id, record FROM r6_sessions WHERE id IN (?,?) ORDER BY id', REGRESSION_SESSIONS).fetchall()
    require({key for key, _ in records} == set(REGRESSION_SESSIONS), 'Required regression sessions missing')
    cases, proofs = [], []
    for session_id, raw in records:
        session = metrics.parse_json(raw)
        require(session['id'] == session_id and session['history_policy'] == 'independent_model_histories_v1',
                'Regression history identity changed')
        indexes = (1, 2, 3) if session_id == REGRESSION_SESSIONS[0] else (0, 1)
        require(len(session['rounds']) > max(indexes), 'Regression turns missing')
        for index in indexes:
            turn = session['rounds'][index]
            messages = turn['input_messages']['r6_response']
            require(turn['input_messages_sha256']['r6_response'] == metrics.digest(messages), 'Saved history hash mismatch')
            saved = turn['sides']['r6_response']['turn']
            require(saved['messages'] == messages and saved['answer'] == saved['remote_result']['output'],
                    'Saved regression output mismatch')
            group = 'reused-r6-session:' + session_id
            probe = ('topic_change', 'correction', 'topic_change')[index - 1] if session_id == REGRESSION_SESSIONS[0] else ('fresh', 'followup')[index]
            case = {'id': group + ':' + str(index), 'scenario_id': group, 'source_group_id': group,
                    'domain': 'reused_category_regression', 'input_language': 'en' if session_id == REGRESSION_SESSIONS[1] and index == 1 else 'tw',
                    'output_language': 'en' if session_id == REGRESSION_SESSIONS[1] and index == 1 else 'tw',
                    'mode': 'fixed_shared_history', 'turn': index, 'probe': probe,
                    'messages': messages, 'sources': {}, 'constraints': [],
                    'rubrics': {'category_membership': 'Review requested categories; no native gold vocabulary supplied.',
                                'list_distinctness': 'Review whether requested examples are distinct, not repeated generic category names.',
                                'correction_uptake': 'Review response to the current clarification without copying earlier errors.',
                                'topic_change': 'Review whether the current topic replaces the obsolete category.',
                                'language_adherence': 'Review requested language; AI-authored rubric is not native certification.'}}
            if session_id == REGRESSION_SESSIONS[1]:
                case['constraints'] = [{'id': 'list_distinctness', 'kind': 'list_distinctness',
                                        'expected': {'format': 'numbered_lines', 'field': None, 'count': 3,
                                                     'allowed': [], 'forbidden': [], 'aliases': {}}}]
            cases.append(case)
            proofs.append({'case_id': case['id'], 'session_id': session_id, 'round_id': turn['id'],
                           'messages_sha256': metrics.digest(messages), 'saved_output_sha256': metrics.digest(saved['answer']),
                           'guard_enabled': session['loop_guard'], 'saved_guard': saved['remote_result'].get('loop_guard'),
                           'context_content_characters': sum(len(m['content']) for m in messages)})
    validate_views(cases, [])
    return cases, {'scope': 'REUSED_development_regression_not_fresh', 'cases': proofs,
                   'sessions_sha256': metrics.digest([metrics.parse_json(raw) for _, raw in records])}


def held_content(cases: list[metrics.EvalCase]) -> list[dict]:
    held = {}
    for case in cases:
        group = case['source_group_id']
        texts = list(case['sources'].values())
        texts += [m['content'] for m in case['messages'] if m['role'] == 'user']
        texts.append('\n'.join(m['content'] for m in case['messages'] if m['role'] != 'system'))
        texts.append(texts[-1] + '\nEXPECTED CONSTRAINTS\n' + json.dumps(case['constraints'], sort_keys=True, ensure_ascii=False))
        for text in texts:
            key = (group, overlap_normalized(text))
            if key not in held:
                held[key] = {'id': group + ':held:' + metrics.digest(text)[:20], 'source_id': group,
                             'source_group_id': group, 'text': text, 'case_ids': [], 'training_eligible': False}
            if case['id'] not in held[key]['case_ids']:
                held[key]['case_ids'].append(case['id'])
    return sorted(held.values(), key=lambda record: record['id'])


def file_sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path: Path):
    return metrics.parse_json(path.read_text(encoding='utf-8'))


def json_lines(path: Path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()), 'Blank JSONL row')
            yield metrics.parse_json(line)


def safe_file(root: Path, name: str) -> Path:
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path')
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Missing/linked artifact')
    return path


def r6_overlap(pack: Path, expected_manifest_sha256: str, held: list[dict]) -> dict:
    require(metrics.is_sha(expected_manifest_sha256) and file_sha(pack / 'manifest.json') == expected_manifest_sha256,
            'R6 manifest hash mismatch')
    manifest = read(pack / 'manifest.json')
    require(read(pack / 'PREPARED.json') == {'manifest_sha256': expected_manifest_sha256}, 'R6 preparation seal mismatch')
    names = {'source-witnesses.jsonl'} | {s['records_path'] for s in manifest['dataset_snapshots']}
    expected_counts = {s['records_path']: s['record_count'] for s in manifest['dataset_snapshots']}
    for snapshot in manifest['dataset_snapshots']:
        require(manifest['artifacts'].get(snapshot['records_path']) == snapshot['records_sha256'], 'Unbound R6 source snapshot')
    index = HoldoutIndex([(record['id'], record['text']) for record in held])
    groups = {record['source_group_id'] for record in held}
    matches, scanned, references = [], {}, {}
    for name in sorted(names):
        path = safe_file(pack, name)
        require(file_sha(path) == manifest['artifacts'].get(name), 'R6 source artifact changed: ' + name)
        references[name] = manifest['artifacts'][name]
        count = 0
        for count, record in enumerate(json_lines(path), 1):
            texts = []
            for field in ('messages', 'english_messages', 'original_english_messages'):
                messages = record.get(field, [])
                texts.extend(m['content'] for m in messages if m['role'] != 'system')
                if messages:
                    texts.append('\n'.join(m['content'] for m in messages if m['role'] != 'system'))
            if isinstance(record.get('text'), str):
                texts.append(record['text'])
            for text in dict.fromkeys(texts):
                match = index.match(text)
                if match:
                    matches.append({'r6_file': name, 'r6_record_id': record.get('id'), **match})
            metadata = record.get('metadata', {})
            identities = [record.get('id'), record.get('source_id')]
            if isinstance(metadata, dict):
                identities += [metadata.get(key) for key in ('group_id', 'family_id', 'source_group_id', 'original_group_id')]
            if any(identity in groups for identity in identities if isinstance(identity, str)):
                matches.append({'r6_file': name, 'r6_record_id': record.get('id'), 'kind': 'source_group_identity'})
        scanned[name] = count
        require(name not in expected_counts or count == expected_counts[name], 'R6 source count mismatch')
    require(scanned['source-witnesses.jsonl'] == sum(manifest['counts'].values()), 'R6 witness coverage mismatch')
    require(not matches, 'Fresh evaluation overlaps the R6 checkpack: ' + json.dumps(matches[:3]))
    return {'r6_manifest_sha256': expected_manifest_sha256, 'verified_artifacts': references, 'scanned_records': scanned,
            'protected_records_scanned': len(held), 'matches': matches,
            'checks': ['normalized_exact', '8_word_shingle_jaccard_at_least_0.85', 'source_group_identity'],
            'scope': 'available_R6_source_snapshots_and_consumed_view_witnesses',
            'freshness_claim': 'AI_authored_development_relative_to_scanned_R6_pack',
            'semantic_overlap_clearance': False, 'foundation_pretraining_overlap': 'unknown',
            'untouched_final_test': False}


def counts(fixed, scenarios, reused) -> dict:
    fresh = all_cases(fixed, scenarios)
    return {'fresh_source_groups': len({c['source_group_id'] for c in fresh}),
            'semantic_task_families': len({c['domain'] for c in fresh}),
            'fixed_views': len(fixed), 'rollout_scenarios': len(scenarios),
            'rollout_turn_views': sum(len(s['expected_per_turn']) for s in scenarios),
            'fresh_derived_test_views': len(fresh), 'reused_fixed_views': len(reused),
            'reused_source_groups': len({c['source_group_id'] for c in reused}),
            'fresh_by_language': dict(sorted(Counter(c['input_language'] for c in fresh).items())),
            'fresh_by_probe': dict(sorted(Counter(c['probe'] for c in fresh).items())),
            'fresh_by_family': dict(sorted(Counter(c['domain'] for c in fresh).items()))}


def encoded(value, *, lines=False) -> bytes:
    if lines:
        text = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n' for row in value)
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    return text.encode('utf-8')


def build(out: Path, *, r6_pack: Path, expected_r6_manifest_sha256: str, regression_db: Path) -> dict:
    require(not out.exists(), 'Output already exists; never overwrite a frozen pack')
    sources = source_scenarios()
    fixed, scenarios = build_views(sources)
    reused, regression_proof = regression_cases(regression_db)
    protected_fresh = held_content(all_cases(fixed, scenarios))
    audit = r6_overlap(r6_pack, expected_r6_manifest_sha256, protected_fresh)
    protected_all = protected_fresh + held_content(reused)
    payloads = {'cases.jsonl': encoded(fixed, lines=True), 'scenarios.jsonl': encoded(scenarios, lines=True),
                'reused-regressions.jsonl': encoded(reused, lines=True), 'source-scenarios.json': encoded(sources),
                'held-content.jsonl': encoded(protected_all, lines=True), 'r6-overlap.json': encoded(audit),
                'regression-provenance.json': encoded(regression_proof)}
    manifest = {'version': VERSION, 'scope': 'AI_authored_development_not_untouched_final_test',
                'training_eligible': False, 'native_gold': False, 'clinical_gold': False,
                'model_evaluation_executed': False, 'execution_location': 'Modal_only',
                'counts': counts(fixed, scenarios, reused),
                'groups_frozen_before_r7_training': sorted({c['source_group_id'] for c in all_cases(fixed, scenarios) + reused}),
                'group_policy': 'All language/history/probe views and future derivatives stay held out together; no train/test prompt-only split.',
                'r6_manifest_sha256': expected_r6_manifest_sha256,
                'code': {name: file_sha(Path(__file__).with_name(name)) for name in (
                    'build_r7_behavior_eval.py', 'r7_eval_metrics.py', 'prepare_understanding_run.py')},
                'artifacts': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())},
                'remote_contract': {'fixed': ['id', 'messages'], 'rollout': ['id', 'system_message', 'user_turns'],
                                    'never_send': ['constraints', 'rubrics', 'expected_per_turn', 'source-scenarios.json']},
                'limitations': ['AI-authored Twi requires independent native-language review.',
                                'Eight authored semantic roots are not a population sample.',
                                'Lexical R6 separation does not establish semantic/pretraining clearance.',
                                'Exact answers/citation IDs do not establish naturalness or broad source entailment.',
                                'Fixed correction histories intentionally include a wrong assistant answer.']}
    payloads['manifest.json'] = encoded(manifest)
    payloads['FROZEN.json'] = encoded({'manifest_sha256': hashlib.sha256(payloads['manifest.json']).hexdigest()})
    out.mkdir(parents=True)
    for name, data in payloads.items():
        with (out / name).open('xb') as stream:
            stream.write(data)
    return verify(out, expected_manifest_sha256=file_sha(out / 'manifest.json'))


def verify(out: Path, *, expected_manifest_sha256: str) -> dict:
    require(metrics.is_sha(expected_manifest_sha256) and file_sha(out / 'manifest.json') == expected_manifest_sha256,
            'Evaluation manifest hash mismatch')
    require(read(out / 'FROZEN.json') == {'manifest_sha256': expected_manifest_sha256}, 'Evaluation seal mismatch')
    manifest = read(out / 'manifest.json')
    require(manifest['version'] == VERSION and manifest['training_eligible'] is False
            and manifest['native_gold'] is False and manifest['clinical_gold'] is False, 'Invalid evaluation scope')
    for name, checksum in manifest['code'].items():
        require(Path(name).name == name and file_sha(Path(__file__).with_name(name)) == checksum, 'Evaluation code changed')
    require({path.name for path in out.iterdir()} == set(manifest['artifacts']) | {'manifest.json', 'FROZEN.json'},
            'Missing/unexpected evaluation artifacts')
    for name, checksum in manifest['artifacts'].items():
        require(file_sha(safe_file(out, name)) == checksum, 'Evaluation artifact changed: ' + name)
    fixed, scenarios = list(json_lines(out / 'cases.jsonl')), list(json_lines(out / 'scenarios.jsonl'))
    reused = list(json_lines(out / 'reused-regressions.jsonl'))
    sources = read(out / 'source-scenarios.json')
    validate_sources(sources)
    expected_fixed, expected_scenarios = build_views(sources)
    require(fixed == expected_fixed and scenarios == expected_scenarios, 'Frozen cases differ from source scenarios')
    validate_views(reused, [])
    require(manifest['counts'] == counts(fixed, scenarios, reused), 'Derived view/group counts differ')
    require(list(json_lines(out / 'held-content.jsonl')) == held_content(all_cases(fixed, scenarios)) + held_content(reused),
            'Protection coverage differs')
    return {'manifest_sha256': expected_manifest_sha256, 'counts': manifest['counts'],
            'scope': manifest['scope'], 'verified': True, 'model_evaluation_executed': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--r6-pack', type=Path)
    parser.add_argument('--expected-r6-manifest-sha256')
    parser.add_argument('--regression-db', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--expected-manifest-sha256')
    args = parser.parse_args()
    if args.verify:
        result = verify(args.out, expected_manifest_sha256=args.expected_manifest_sha256)
    else:
        require(args.r6_pack is not None and args.regression_db is not None, 'R6 pack and regression DB required')
        result = build(args.out, r6_pack=args.r6_pack, expected_r6_manifest_sha256=args.expected_r6_manifest_sha256,
                       regression_db=args.regression_db)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
