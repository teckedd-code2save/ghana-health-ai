#!/usr/bin/env python3
"""Prepare offline, explicitly launch detached R6, or collect raw partial checkpoints."""
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'modal/train/r7_behavior_baseline.py'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_worker(path=WORKER, expected_sha=None):
    path = Path(path)
    source = path.read_bytes()
    checksum = hashlib.sha256(source).hexdigest()
    if expected_sha is not None and checksum != expected_sha:
        raise ValueError('Worker hash differs before import')
    name = '_r7_baseline_worker_' + checksum[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def prepare(out, evaluation, evaluation_sha, bundle, run_id):
    import build_r7_behavior_eval as inputs
    worker = load_worker()
    out, evaluation, bundle = Path(out), Path(evaluation).resolve(), Path(bundle).resolve()
    worker.require(not out.exists(), 'Use a new package directory')
    worker.require(not out.resolve().is_relative_to(evaluation), 'Do not write inside the frozen evaluation pack')
    inputs.verify(evaluation, expected_manifest_sha256=evaluation_sha)
    manifest = worker.verify_bundle(bundle, worker.EXPECTED)
    fixed = list(inputs.json_lines(evaluation / 'cases.jsonl'))
    reused = list(inputs.json_lines(evaluation / 'reused-regressions.jsonl'))
    scenarios = list(inputs.json_lines(evaluation / 'scenarios.jsonl'))
    plan = {'version': worker.PLAN_VERSION, 'cases': [inputs.remote_fixed(case) for case in fixed + reused],
            'rollouts': [inputs.remote_rollout(scenario) for scenario in scenarios]}
    slots = []
    for scope, cases in (('fresh', fixed), ('reused', reused)):
        slots.extend({'id': case['id'], 'case_id': case['id'], 'scope': scope, 'kind': 'fixed',
                      'source_group_id': case['source_group_id'], 'language': case['input_language'], 'probe': case['probe']}
                     for case in cases)
    for scenario in scenarios:
        for turn, case in enumerate(scenario['expected_per_turn']):
            slots.append({'id': worker.rollout_slot_id(scenario['id'], turn), 'case_id': case['id'],
                          'scope': 'fresh', 'kind': 'rollout', 'source_group_id': case['source_group_id'],
                          'language': case['input_language'], 'probe': case['probe']})
    source_sha = file_sha(WORKER)
    binding = worker.binding_for(run_id, plan, manifest, worker.EXPECTED, source_sha, evaluation_sha)
    counts = {'fixed_fresh': len(fixed), 'fixed_reused': len(reused), 'rollout_scenarios': len(scenarios),
              'rollout_turns': sum(len(s['user_turns']) for s in scenarios), 'expected_slots': len(slots)}
    out.mkdir(parents=True)
    (out / 'worker.py').write_bytes(WORKER.read_bytes())
    worker.save(out / 'inputs.json', plan)
    worker.save(out / 'slots.json', slots)
    execution = {'version': worker.VERSION, 'binding': binding, 'bundle': str(bundle),
                 'evaluation': str(evaluation), 'counts': counts,
                 'files': {name: file_sha(out / name) for name in ('worker.py', 'inputs.json', 'slots.json')},
                 'launcher_sha256': file_sha(__file__), 'detached': True, 'gpu': 'H100',
                 'app_name': worker.APP_NAME, 'volume': worker.VOLUME, 'live_ui_changed': False}
    worker.save(out / 'execution.json', execution)
    checksum = file_sha(out / 'execution.json')
    worker.save(out / 'PREPARED.json', {'execution_sha256': checksum})
    verify_package(out, checksum)
    return {'package': str(out.resolve()), 'execution_sha256': checksum, 'counts': counts, 'gpu_launched': False}


def verify_package(package, expected_sha):
    package = Path(package)
    # Check the independent execution seal and worker bytes before importing staged code.
    if not isinstance(expected_sha, str) or file_sha(package / 'execution.json') != expected_sha:
        raise ValueError('Execution package hash mismatch')
    execution = json.loads((package / 'execution.json').read_text(encoding='utf-8'))
    if set(execution['files']) != {'worker.py', 'inputs.json', 'slots.json'}:
        raise ValueError('Unexpected packaged artifacts')
    for name, checksum in execution['files'].items():
        path = package / name
        if path.is_symlink() or not path.is_file() or file_sha(path) != checksum:
            raise ValueError('Packaged artifact changed: ' + name)
    worker = load_worker(package / 'worker.py', execution['files']['worker.py'])
    worker.require(worker.read(package / 'PREPARED.json') == {'execution_sha256': expected_sha}, 'Preparation seal mismatch')
    worker.require({p.name for p in package.iterdir()} == set(execution['files']) | {'execution.json', 'PREPARED.json'},
                   'Unbound package file; store receipts/collections outside the package')
    worker.require(execution['version'] == worker.VERSION and execution['gpu'] == 'H100'
                   and execution['app_name'] == worker.APP_NAME and execution['detached'] is True,
                   'Wrong standalone baseline resources')
    plan, slots = worker.read(package / 'inputs.json'), worker.read(package / 'slots.json')
    worker.require(worker.validate(plan) == execution['binding']['plan_sha256']
                   and execution['binding']['worker_sha256'] == execution['files']['worker.py']
                   and execution['binding']['bundle_sha256'] == worker.EXPECTED
                   and execution['binding']['guard'] == 'raw', 'Execution binding differs')
    expected_ids = {slot['id'] for slot in worker.iter_slots(plan, {})}
    worker.require(len(slots) == len({slot['id'] for slot in slots}) == len(expected_ids)
                   and {slot['id'] for slot in slots} == expected_ids
                   and execution['counts']['expected_slots'] == len(slots), 'Slot mapping/count differs')
    return worker, execution, plan, slots


@contextmanager
def receipt_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + '.lock')
    if lock.is_symlink():
        raise ValueError('Linked receipt lock')
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another launcher is using this receipt') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class ModalBackend:
    def __init__(self, profile, environment):
        os.environ['MODAL_PROFILE'] = profile
        os.environ['MODAL_ENVIRONMENT'] = environment
        import modal
        self.modal = modal

    def application(self, worker, execution):
        return worker.create_app(execution['bundle'], worker.EXPECTED, modal_module=self.modal)

    def output(self):
        return self.modal.enable_output()

    def claim(self, worker, run_id, attempt):
        try:
            return self.modal.Dict.from_name(worker.CLAIMS, create_if_missing=False).get(worker.claim_key(run_id, attempt))
        except self.modal.exception.NotFoundError:
            return None

    def remember(self, worker, run_id, attempt, call_id, binding):
        claims = self.modal.Dict.from_name(worker.CLAIMS, create_if_missing=True)
        value = {'call_id': call_id, 'binding': binding, 'attempt': attempt}
        claims.put(worker.claim_key(run_id, attempt), value, skip_if_exists=True)
        worker.require(claims.get(worker.claim_key(run_id, attempt)) == value, 'Another call owns this attempt')

    def graph(self, call_id):
        return self.modal.FunctionCall.from_id(call_id).get_call_graph()

    def progress(self, worker, run_id):
        try:
            volume = self.modal.Volume.from_name(worker.VOLUME, create_if_missing=False)
            return worker.parse(b''.join(volume.read_file('/' + worker.PREFIX + '/' + run_id + '/progress.json')))
        except (FileNotFoundError, self.modal.exception.NotFoundError):
            return None


def reconcile(worker, receipt, backend):
    if not receipt['attempts']:
        return None
    attempt = receipt['attempts'][-1]
    claim = backend.claim(worker, receipt['binding']['run_id'], attempt['attempt'])
    if claim is not None:
        worker.require(claim['binding'] == receipt['binding'] and claim['attempt'] == attempt['attempt'], 'Invocation claim differs')
        worker.require(attempt.get('call_id') in (None, claim['call_id']), 'Call identity differs')
        attempt.update(call_id=claim['call_id'], state='submitted', spawn_started=True)
    return attempt


def launch(package, execution_sha, receipt_path, *, authorize_gpu=False, resume=False,
           profile='createdliving1000', environment='main', backend=None):
    worker, execution, plan, _ = verify_package(package, execution_sha)
    worker.require(authorize_gpu, 'Explicit --authorize-gpu required; no GPU call made')
    worker.require(file_sha(__file__) == execution['launcher_sha256'], 'Launcher changed since preparation')
    interpreter = worker.require_launch_python()
    worker.verify_bundle(execution['bundle'], worker.EXPECTED)
    receipt_path = Path(receipt_path)
    worker.require(not receipt_path.resolve().is_relative_to(Path(package).resolve()), 'Receipt must be outside the package')
    with receipt_lock(receipt_path):
        if receipt_path.exists():
            worker.require(resume, 'Existing receipt; collect or explicitly --resume, never duplicate')
            receipt = worker.read(receipt_path)
            worker.require(receipt['execution_sha256'] == execution_sha and receipt['binding'] == execution['binding']
                           and receipt['profile'] == profile and receipt['environment'] == environment, 'Receipt identity differs')
        else:
            worker.require(not resume, 'No receipt to resume')
            receipt = {'version': worker.VERSION, 'execution_sha256': execution_sha,
                       'binding': execution['binding'], 'profile': profile, 'environment': environment,
                       'detached': True, 'state': 'prepared', 'attempts': [], 'launch_python': interpreter}
            worker.save(receipt_path, receipt)
        backend = backend or ModalBackend(profile, environment)
        previous = reconcile(worker, receipt, backend)
        worker.save(receipt_path, receipt)
        attempt_number = 0
        if previous is not None:
            if previous.get('call_id'):
                worker.terminal_state(backend.graph(previous['call_id']))
                progress = backend.progress(worker, receipt['binding']['run_id'])
                if progress is not None:
                    worker.verify_report(progress, plan, receipt['binding'])
                    if progress['state'] == 'complete':
                        receipt['state'] = 'complete'
                        worker.save(receipt_path, receipt)
                        return receipt
                # Register known calls even if their container failed before its first claim.
                backend.remember(worker, receipt['binding']['run_id'], previous['attempt'], previous['call_id'], receipt['binding'])
                attempt_number = previous['attempt'] + 1
            else:
                worker.require(not previous['spawn_started'], 'Submission outcome unknown; collect/recover claim before any resubmission')
                attempt_number = previous['attempt']
        else:
            worker.require(backend.claim(worker, receipt['binding']['run_id'], 0) is None,
                           'Run ID already has an invocation; recover the original receipt')
        attempt = {'attempt': attempt_number, 'state': 'preparing', 'spawn_started': False,
                   'call_id': None, 'launch_python': interpreter}
        receipt['attempts'].append(attempt)
        receipt['state'] = 'preparing'
        worker.save(receipt_path, receipt)
        try:
            app, function, _ = backend.application(worker, execution)
            with backend.output(), app.run(detach=True):
                attempt.update(state='submitting', spawn_started=True)
                receipt['state'] = 'submitting'
                worker.save(receipt_path, receipt)
                call = function.spawn(receipt['binding']['run_id'], plan, receipt['binding'], attempt_number)
                attempt.update(state='submitted', call_id=call.object_id)
                receipt.update(state='submitted', app_id=app.app_id)
                worker.save(receipt_path, receipt)
                backend.remember(worker, receipt['binding']['run_id'], attempt_number, call.object_id, receipt['binding'])
        except Exception as error:
            attempt['error'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            receipt['state'] = 'submitted' if attempt['call_id'] else 'submission_unknown' if attempt['spawn_started'] else 'setup_failed'
            worker.save(receipt_path, receipt)
            raise
        return receipt


def collect(package, receipt_path, out, *, profile='createdliving1000', environment='main', backend=None):
    receipt_path, out = Path(receipt_path), Path(out)
    with receipt_lock(receipt_path):
        # Receipt was persisted before submission and supplies the independent package seal.
        initial = json.loads(receipt_path.read_text(encoding='utf-8'))
        worker, execution, plan, slots = verify_package(package, initial['execution_sha256'])
        receipt = worker.read(receipt_path)
        worker.require(receipt['binding'] == execution['binding'] and receipt['profile'] == profile
                       and receipt['environment'] == environment, 'Collection identity differs')
        worker.require(not out.resolve().is_relative_to(Path(package).resolve())
                       and not out.resolve().is_relative_to(Path(execution['evaluation']).resolve()), 'Do not overwrite frozen inputs')
        backend = backend or ModalBackend(profile, environment)
        attempt = reconcile(worker, receipt, backend)
        worker.save(receipt_path, receipt)
        progress = backend.progress(worker, receipt['binding']['run_id'])
        if progress is None:
            return {'state': 'no_checkpoint_yet', 'receipt_state': receipt['state'],
                    'call_id': attempt.get('call_id') if attempt else None, 'expected_slots': len(slots)}
        worker.verify_report(progress, plan, receipt['binding'])
        snapshot = progress['snapshot_sha256']
        target = out / 'snapshots' / (snapshot + '.json')
        if not target.exists():
            worker.save(target, progress)
        latest = out / 'latest.json'
        if latest.exists():
            old = worker.verify_report(worker.read(latest), plan, receipt['binding'])
            current = {r['id']: r for r in progress['results']}
            worker.require(all(current.get(r['id']) == r for r in old['results']), 'Checkpoint regressed or changed finalized output')
        worker.save(latest, progress)
        done = {r['id']: r for r in progress['results']}
        scopes = {}
        for scope in ('fresh', 'reused'):
            selected = [s for s in slots if s['scope'] == scope]
            states = {state: 0 for state in (*worker.FINAL_STATES, 'pending')}
            for slot in selected:
                states[done[slot['id']]['state'] if slot['id'] in done else 'pending'] += 1
            scopes[scope] = {'expected_slots': len(selected), 'states': states}
        summary = {'state': progress['state'], 'counts': progress['counts'], 'scopes': scopes,
                   'snapshot_sha256': snapshot, 'raw_results': str(latest.resolve()),
                   'pending_slot_ids': [s['id'] for s in slots if s['id'] not in done],
                   'semantic_quality_established': False, 'clinical_certification': False}
        worker.save(out / 'collection.json', summary)
        if progress['state'] == 'complete':
            receipt['state'] = 'complete'
            worker.save(receipt_path, receipt)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'preflight', 'launch', 'collect'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--evaluation-sha')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--expected-execution-sha')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--profile', default='createdliving1000')
    parser.add_argument('--environment', default='main')
    parser.add_argument('--authorize-gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if not all((args.evaluation, args.evaluation_sha, args.bundle, args.run_id)):
            parser.error('prepare requires --evaluation, --evaluation-sha, --bundle and --run-id')
        result = prepare(args.package, args.evaluation, args.evaluation_sha, args.bundle, args.run_id)
    elif args.command in ('verify', 'preflight'):
        worker, execution, _, _ = verify_package(args.package, args.expected_execution_sha)
        result = {'execution_sha256': args.expected_execution_sha, 'counts': execution['counts'], 'gpu_launched': False}
        if args.command == 'preflight':
            result['launch_python'] = worker.require_launch_python()
            worker.require(file_sha(__file__) == execution['launcher_sha256'], 'Launcher changed since preparation')
            worker.verify_bundle(execution['bundle'], worker.EXPECTED)
    elif args.command == 'launch':
        if not args.receipt:
            parser.error('launch requires --receipt')
        result = launch(args.package, args.expected_execution_sha, args.receipt, authorize_gpu=args.authorize_gpu,
                        resume=args.resume, profile=args.profile, environment=args.environment)
    else:
        if not args.receipt or not args.out:
            parser.error('collect requires --receipt and --out')
        result = collect(args.package, args.receipt, args.out, profile=args.profile, environment=args.environment)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
