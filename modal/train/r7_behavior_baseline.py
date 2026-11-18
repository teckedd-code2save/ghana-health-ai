"""Detached raw R6 baseline; importing this file neither contacts Modal nor loads a model."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

VERSION = 'r7-behavior-baseline-v1'
PLAN_VERSION = 'r7-behavior-inputs-v1'
EXPECTED = '529200bb9482784bf7045985de3ffda7a712055978ec9f302f19dfeec4ed670a'
APP_NAME = 'ghana-r7-behavior-baseline-private'
VOLUME = 'ghana-health-understand-train'
CACHE = 'ghana-health-understanding-hf-cache'
CLAIMS = 'ghana-r7-behavior-invocations'
PREFIX = 'understanding-adaptation-r7/behavior'
IMAGE_PYTHON = '3.11'
FINAL_STATES = ('complete', 'incomplete', 'invalid', 'error', 'blocked')
TERMINAL_CALLS = {'SUCCESS', 'FAILURE', 'INIT_FAILURE', 'TERMINATED', 'TIMEOUT'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def parse(text):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(text, object_pairs_hook=unique,
                        parse_constant=lambda _: require(False, 'Nonfinite JSON'))
    digest(result)
    return result


def read(path):
    return parse(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    require(not path.is_symlink(), 'Linked output file')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    require(not temporary.is_symlink(), 'Linked temporary output')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_file(root, name):
    root, relative = Path(root), Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path')
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            'Missing or linked artifact: ' + name)
    return path


def verify_bundle(bundle, expected_sha):
    """Hash every executable dependency before any bundle code can be imported."""
    bundle = Path(bundle)
    require(is_sha(expected_sha) and sha(safe_file(bundle, 'manifest.json')) == expected_sha,
            'Expected sealed R6 bundle')
    require(read(safe_file(bundle, 'SEALED.json')) == {'manifest_sha256': expected_sha}, 'Bundle seal mismatch')
    manifest = read(bundle / 'manifest.json')
    require(manifest['version'] == 'r6-comparison-inference-v1' and manifest['training'] is False
            and manifest['private'] is True, 'Not a private R6 inference bundle')
    require(not any(path.is_symlink() for path in bundle.rglob('*')), 'Linked bundle artifact')
    names = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    require(names == set(manifest['files']) | {'manifest.json', 'SEALED.json'}, 'Unbound or missing bundle file')
    require({'code/r6_inference.py', 'code/r2_inference.py', 'code/afrique_instruction_baseline_core.py',
             'code/understanding_r2_demo.py'} <= set(manifest['files']), 'Missing frozen runtime helpers')
    for name, checksum in manifest['files'].items():
        require(is_sha(checksum) and sha(safe_file(bundle, name)) == checksum, 'Bundle artifact changed: ' + name)
    return manifest


def validate_messages(messages):
    require(type(messages) is list and 1 <= len(messages) <= 129, 'Bounded messages required')
    require(all(type(m) is dict and set(m) == {'role', 'content'} for m in messages), 'Invalid message fields')
    start = int(messages[0]['role'] == 'system')
    require((len(messages) - start) % 2 == 1, 'Messages must end in user')
    for index, message in enumerate(messages):
        expected = 'system' if start and index == 0 else 'user' if (index - start) % 2 == 0 else 'assistant'
        require(message['role'] == expected and isinstance(message['content'], str)
                and message['content'].strip() and len(message['content']) <= 8000, 'Invalid message content/order')
    require(sum(len(m['content']) for m in messages) <= 65536, 'Conversation exceeds bound')


def validate_run_id(run_id):
    require(isinstance(run_id, str) and re.fullmatch(r'r7-baseline-[a-z0-9-]{4,80}', run_id), 'Invalid run ID')


def rollout_slot_id(scenario_id, turn):
    return scenario_id + ':turn-' + str(turn)


def validate(plan):
    require(type(plan) is dict and set(plan) == {'version', 'cases', 'rollouts'}
            and plan['version'] == PLAN_VERSION, 'Only model-visible cases and rollouts accepted')
    require(type(plan['cases']) is list and type(plan['rollouts']) is list
            and 1 <= len(plan['cases']) + len(plan['rollouts']) <= 512, 'Bounded nonempty plan required')
    roots, slots = set(), set()
    for kind, rows in (('fixed', plan['cases']), ('rollout', plan['rollouts'])):
        for row in rows:
            fields = {'id', 'messages'} if kind == 'fixed' else {'id', 'system_message', 'user_turns'}
            require(type(row) is dict and set(row) == fields, 'Unexpected non-model-visible fields')
            require(isinstance(row['id'], str) and 0 < len(row['id']) <= 300 and row['id'].strip()
                    and row['id'] not in roots, 'Duplicate or invalid input ID')
            roots.add(row['id'])
            if kind == 'fixed':
                validate_messages(row['messages'])
                identities = [row['id']]
            else:
                require(isinstance(row['system_message'], str) and row['system_message'].strip()
                        and len(row['system_message']) <= 8000, 'Rollout system message required')
                require(type(row['user_turns']) is list and 2 <= len(row['user_turns']) <= 6
                        and all(isinstance(q, str) and q.strip() and len(q) <= 8000 for q in row['user_turns']),
                        'Invalid rollout user turns')
                identities = [rollout_slot_id(row['id'], turn) for turn in range(len(row['user_turns']))]
            require(not slots.intersection(identities), 'Colliding fixed/rollout slot IDs')
            slots.update(identities)
    require(len(slots) <= 2048, 'Too many planned response slots')
    return digest(plan)


def binding_for(run_id, plan, manifest, bundle_sha, worker_sha, evaluation_sha):
    validate_run_id(run_id)
    require(all(is_sha(value) for value in (bundle_sha, worker_sha, evaluation_sha)), 'Missing execution hashes')
    protocol = {'decoding': manifest['decoding'], 'runtime': manifest['runtime'], 'guard': 'raw',
                'chat_template_sha256': manifest['files']['tokenizer/chat_template.jinja'],
                'evaluation_manifest_sha256': evaluation_sha}
    return {'version': VERSION, 'run_id': run_id, 'bundle_sha256': bundle_sha,
            'plan_sha256': validate(plan), 'worker_sha256': worker_sha, 'variant': 'r6_response',
            'decoding': manifest['decoding'], 'guard': 'raw', 'training': False,
            'evaluation_manifest_sha256': evaluation_sha, 'protocol_sha256': digest(protocol)}


def iter_slots(plan, done):
    for row in plan['cases']:
        yield {'id': row['id'], 'kind': 'fixed', 'scenario_id': None, 'turn': 0,
               'messages': row['messages'], 'blocked_by': None}
    for row in plan['rollouts']:
        history = [{'role': 'system', 'content': row['system_message']}]
        blocked = None
        for turn, question in enumerate(row['user_turns']):
            identifier = rollout_slot_id(row['id'], turn)
            messages = history + [{'role': 'user', 'content': question}] if history is not None and blocked is None else []
            yield {'id': identifier, 'kind': 'rollout', 'scenario_id': row['id'], 'turn': turn,
                   'messages': messages, 'blocked_by': blocked}
            record = done.get(identifier)
            if record is None:
                history = None
            elif record['state'] == 'complete' and record['verified'] and record['output'].strip():
                history = messages + [{'role': 'assistant', 'content': record['output']}]
            else:
                blocked = blocked or identifier


def count_report(plan, results):
    expected = len(plan['cases']) + sum(len(row['user_turns']) for row in plan['rollouts'])
    states = Counter(row['state'] for row in results)
    return {'expected_slots': expected, 'finalized_slots': len(results), 'pending_slots': expected - len(results),
            'states': {**{state: states[state] for state in FINAL_STATES}, 'pending': expected - len(results)},
            'verified_slots': sum(row['verified'] for row in results)}


def verify_report(report, plan, binding):
    require(report['binding'] == binding and report['plan_sha256'] == validate(plan), 'Progress binding changed')
    require(report['snapshot_sha256'] == digest({k: v for k, v in report.items() if k != 'snapshot_sha256'}),
            'Progress snapshot hash mismatch')
    require(report['state'] in ('loading', 'running', 'paused', 'error', 'complete'), 'Unknown progress state')
    results = report['results']
    require(type(results) is list and len({r['id'] for r in results}) == len(results), 'Duplicate progress records')
    done = {record['id']: record for record in results}
    allowed = set()
    for slot in iter_slots(plan, done):
        allowed.add(slot['id'])
        if slot['id'] not in done:
            continue
        record = done[slot['id']]
        require(all(record[k] == slot[k] for k in slot), 'Saved input/history/slot changed')
        require(record['state'] in FINAL_STATES and type(record['verified']) is bool
                and isinstance(record['output'], str), 'Invalid saved result state')
        require(bool(record['messages']) or record['state'] == 'blocked', 'Result without a request history')
        require((record['state'] == 'blocked') == (slot['blocked_by'] is not None), 'Invalid downstream blocking')
        if record['state'] in ('complete', 'incomplete', 'invalid'):
            require(record['verified'] and type(record['result']) is dict
                    and type(record['result'].get('complete')) is bool
                    and record['output'] == record['result']['output'], 'Missing verified native result')
            expected_state = 'invalid' if not record['output'].strip() else 'complete' if record['result']['complete'] else 'incomplete'
            require(record['state'] == expected_state, 'Saved completion state differs from native result')
    require(set(done) <= allowed and report['counts'] == count_report(plan, results), 'Progress inventory/counts changed')
    require(report['state'] != 'complete' or report['counts']['pending_slots'] == 0, 'Incomplete finalized report')
    if report.get('inflight') is not None:
        require(report['inflight']['id'] in allowed - set(done), 'Invalid inflight slot')
    return report


class InvalidNativeResult(ValueError):
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def load_runtime(bundle, expected_sha):
    verify_bundle(bundle, expected_sha)
    code = Path(bundle).resolve() / 'code'
    sys.dont_write_bytecode = True
    # Sibling imports must never resolve to a cached module from another bundle.
    for path in code.glob('*.py'):
        cached = sys.modules.get(path.stem)
        if cached is not None and Path(getattr(cached, '__file__', '')).resolve() != path:
            del sys.modules[path.stem]
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('_r7_frozen_r6', code / 'r6_inference.py')
    inference = importlib.util.module_from_spec(spec)
    exec(compile((code / 'r6_inference.py').read_bytes(), str(code / 'r6_inference.py'), 'exec'), inference.__dict__)
    manifest = inference.load_bundle(bundle, expected_sha)
    from afrique_instruction_baseline_core import prompt_ids
    from understanding_r2_demo import validate_reply
    runtime = inference.Runtime(bundle, expected_sha)

    class Adapter:
        def reply(self, messages, on_text):
            inference.validate_messages(messages)
            ids = prompt_ids(runtime.tokenizer, messages, {'context_tokens': 4096, 'max_new_tokens': 256})
            result = runtime.reply(messages, 'r6_response', on_text=on_text)
            try:
                validate_reply(result, runtime.identity('r6_response'), messages, manifest['decoding'], result['output'], False)
                require(result['prompt_ids_sha256'] == digest(ids), 'Native tokenized request differs')
            except Exception as error:
                raise InvalidNativeResult(str(error), result) from error
            return result
    return Adapter()


def evaluate_local(root, plan, binding, runtime_factory, *, commit=lambda: None,
                   attempt=0, call_id='local-test', seconds=6900, clock=time.monotonic):
    """CPU-testable execution engine; runtime_factory is the model-loading boundary."""
    validate(plan)
    require(binding['plan_sha256'] == digest(plan) and binding['guard'] == 'raw', 'Wrong engine binding')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    require(not root.is_symlink(), 'Linked progress directory')
    path = root / 'progress.json'
    tick = clock()
    report = {'binding': binding, 'plan_sha256': digest(plan), 'results': [], 'state': 'loading',
              'attempt': attempt, 'call_id': call_id, 'inflight': None, 'error': None,
              'started_at_unix': time.time(), 'clinical_certification': False, 'semantic_quality_established': False}
    if path.exists():
        require(not path.is_symlink(), 'Linked progress file')
        report = verify_report(read(path), plan, binding)
        if report['state'] == 'complete':
            return report
        require(attempt >= report['attempt'], 'Stale invocation cannot overwrite later progress')
    done = {r['id']: r for r in report['results']}

    def checkpoint():
        report['counts'] = count_report(plan, report['results'])
        report['attempt_elapsed_seconds'] = clock() - tick
        report['snapshot_sha256'] = digest({k: v for k, v in report.items() if k != 'snapshot_sha256'})
        save(path, report)
        commit()

    def finish(slot, state, *, result=None, output='', error=None, verified=False):
        record = {**slot, 'state': state, 'result': result, 'output': output, 'error': error, 'verified': verified}
        report['results'].append(record)
        done[slot['id']] = record
        report['inflight'] = None
        checkpoint()

    if report.get('inflight'):
        inflight = report['inflight']
        slot = next(s for s in iter_slots(plan, done) if s['id'] == inflight['id'])
        require(slot['messages'] == inflight['messages'], 'Interrupted request history changed')
        finish(slot, 'error', error={'type': 'InterruptedWithoutResult',
                                    'message': 'Prior attempt ended after request checkpoint; not repeated.'})
    report.update(state='loading', attempt=attempt, call_id=call_id, error=None)
    checkpoint()
    if report['counts']['pending_slots'] == 0:
        report['state'] = 'complete'
        checkpoint()
        return report
    try:
        runtime = runtime_factory()
    except Exception as error:
        report.update(state='error', error={'phase': 'runtime_loading', 'type': type(error).__name__, 'message': str(error)[:2000]})
        checkpoint()
        return report
    report['state'] = 'running'
    checkpoint()
    try:
        for slot in iter_slots(plan, done):
            if slot['id'] in done:
                require(all(done[slot['id']][key] == value for key, value in slot.items()), 'Resumed request changed')
                continue
            if slot['blocked_by'] is not None:
                finish(slot, 'blocked', error={'type': 'UpstreamNotComplete', 'message': 'Earlier rollout turn did not complete.'})
                continue
            if clock() - tick >= seconds:
                report['state'] = 'paused'
                checkpoint()
                return report
            require(bool(slot['messages']), 'Missing preceding rollout record')
            report['inflight'] = {'id': slot['id'], 'messages': slot['messages'], 'attempt': attempt}
            checkpoint()
            chunks = []
            try:
                validate_messages(slot['messages'])
                result = runtime.reply(slot['messages'], on_text=chunks.append)
                require(type(result) is dict and isinstance(result.get('output'), str)
                        and type(result.get('complete')) is bool, 'Malformed runtime result')
                digest(result)
                state = 'invalid' if not result['output'].strip() else 'complete' if result['complete'] else 'incomplete'
                outcome = {'result': result, 'output': result['output'], 'verified': True}
            except Exception as error:
                result = error.result if isinstance(error, InvalidNativeResult) else None
                output = result.get('output', '') if type(result) is dict else ''.join(chunks)
                state = 'error'
                outcome = {'result': result, 'output': output if isinstance(output, str) else '',
                           'error': {'type': type(error).__name__, 'message': str(error)[:2000]}}
            # Persistence failures must not append a second result for the same slot.
            finish(slot, state, **outcome)
        report['state'] = 'complete'
        checkpoint()
    except Exception as error:
        report.update(state='error', error={'phase': 'execution_or_checkpoint', 'type': type(error).__name__, 'message': str(error)[:2000]})
        checkpoint()
        raise
    return report


def terminal_state(graph):
    require(bool(graph), 'No terminal call evidence; no duplicate submission')
    states = []
    for node in graph:
        state = node.status.name
        require(state in TERMINAL_CALLS, 'Prior call may still be active; no duplicate submission')
        states.append(state)
        if node.children:
            states.append(terminal_state(node.children))
    return 'SUCCESS' if all(state == 'SUCCESS' for state in states) else 'FAILED'


def claim_key(run_id, attempt):
    validate_run_id(run_id)
    require(type(attempt) is int and 0 <= attempt <= 100, 'Invalid attempt number')
    return run_id + ':attempt-' + str(attempt)


def claim_attempt(claims, run_id, attempt, owner, binding, call_graph):
    key = claim_key(run_id, attempt)
    if attempt:
        previous = claims.get(claim_key(run_id, attempt - 1))
        require(previous is not None and previous['binding'] == binding, 'Previous invocation identity unavailable')
        terminal_state(call_graph(previous['call_id']))
    claim = {'call_id': owner, 'binding': binding, 'attempt': attempt}
    claims.put(key, claim, skip_if_exists=True)
    require(claims.get(key) == claim, 'Attempt already owned; do not run duplicate GPU work')
    return claim


def remote_evaluate(run_id, plan, binding, attempt, volume, claims):
    import modal
    require(binding['run_id'] == run_id and binding['bundle_sha256'] == EXPECTED
            and binding['worker_sha256'] == sha(__file__), 'Worker execution identity differs')
    owner = modal.current_function_call_id()
    claim_attempt(claims, run_id, attempt, owner, binding,
                  lambda call_id: modal.FunctionCall.from_id(call_id).get_call_graph())
    volume.reload()
    root = Path('/artifacts') / PREFIX / run_id

    def factory():
        manifest = verify_bundle(Path('/bundle'), EXPECTED)
        require(binding == binding_for(run_id, plan, manifest, EXPECTED, sha(__file__), binding['evaluation_manifest_sha256']),
                'Frozen protocol differs')
        return load_runtime(Path('/bundle'), EXPECTED)
    return evaluate_local(root, plan, binding, factory, commit=volume.commit, attempt=attempt, call_id=owner)


def require_launch_python():
    """Serialized functions must use the pinned image's Python minor version."""
    actual = '.'.join(map(str, sys.version_info[:2]))
    require(actual == IMAGE_PYTHON and sys.implementation.name == 'cpython',
            'Baseline launch requires CPython ' + IMAGE_PYTHON + '; current interpreter is '
            + actual + '. Use the matching Python 3.11 launcher environment; do not change the model image.')
    return {'implementation': sys.implementation.name, 'version': sys.version.split()[0],
            'executable': sys.executable, 'image_python': IMAGE_PYTHON}


def create_app(bundle, expected_sha, *, modal_module=None):
    """Only the explicitly authorized launcher calls this resource-definition boundary."""
    require_launch_python()
    require(expected_sha == EXPECTED, 'Only the pinned unchanged R6 bundle is allowed')
    manifest = verify_bundle(bundle, expected_sha)
    if modal_module is None:
        import modal as modal_module
    modal = modal_module
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    cache = modal.Volume.from_name(CACHE, create_if_missing=False)
    claims = modal.Dict.from_name(CLAIMS, create_if_missing=True)
    worker_sha = sha(__file__)
    image = modal.Image.debian_slim(python_version=IMAGE_PYTHON).pip_install(
        *[name + '==' + version for name, version in manifest['runtime'].items()]).env({
            'PYTHONDONTWRITEBYTECODE': '1', 'HF_HOME': '/cache/hf', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'DO_NOT_TRACK': '1',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    image = image.add_local_dir(str(Path(bundle).resolve()), '/bundle')
    image = image.add_local_file(str(Path(__file__).resolve()), '/worker/r7_behavior_baseline.py')

    @app.function(image=image, gpu='H100', cpu=4, memory=49152, timeout=7200, retries=0,
                  max_containers=1, serialized=True, volumes={'/cache': cache.read_only(), '/artifacts': volume})
    def evaluate(run_id, plan, binding, attempt):
        import hashlib
        import importlib.util
        import modal
        from pathlib import Path
        path = Path('/worker/r7_behavior_baseline.py')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != worker_sha:
            raise ValueError('Remote worker source changed')
        spec = importlib.util.spec_from_file_location('_r7_verified_worker', path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(data, str(path), 'exec'), module.__dict__)
        remote_claims = modal.Dict.from_name('ghana-r7-behavior-invocations', create_if_missing=True)
        return module.remote_evaluate(run_id, plan, binding, attempt, volume, remote_claims)

    return app, evaluate, claims
