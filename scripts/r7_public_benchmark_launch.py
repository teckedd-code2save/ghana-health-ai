#!/usr/bin/env python3
"""Offline freeze/prepare; explicit detached launch/resume; read-only collection."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil

import r7_public_benchmark_protocol as core

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'modal/train/r7_public_benchmark.py'


def prepare(frozen, frozen_sha, bundle, bundle_sha, out, run_id):
    frozen, bundle, out = Path(frozen).resolve(), Path(bundle).resolve(), Path(out).absolute()
    core.validate_run_id(run_id)
    core.require(not out.exists() and not out.is_relative_to(frozen) and not out.is_relative_to(bundle),
                 'Use a new execution package outside the frozen inputs/bundle')
    core.verify_frozen(frozen, frozen_sha)
    manifest, _ = core.bundle_module(bundle, bundle_sha)
    original = core.read(frozen / 'historical/manifest.json')
    core.require(manifest['variants']['r6_response'] == original['adapters']['r6'], 'Original R6 adapter differs')
    out.mkdir(parents=True)
    shutil.copytree(frozen, out / 'frozen')
    code = out / 'code'
    code.mkdir()
    for name in ('r7_public_benchmark_protocol.py', 'r7_public_benchmark_protection.py', 'r7_public_benchmark_launch.py',
                 'r6_benchmark_protocol.py'):
        shutil.copyfile(Path(__file__).with_name(name), code / name)
    shutil.copyfile(WORKER, code / 'worker.py')
    binding = {'version': core.VERSION, 'run_id': run_id, 'frozen_manifest_sha256': frozen_sha,
               'inputs_sha256': core.INPUTS_SHA, 'bundle_sha256': bundle_sha,
               'model': manifest['model'], 'revision': manifest['revision'], 'base_sha256': manifest['base_sha256'],
               'adapters': {v: manifest['variants'][v] for v in core.VARIANTS}, 'runtime': manifest['runtime'],
               'tokenizer_files': {k: v for k, v in manifest['files'].items() if k.startswith('tokenizer/')},
               'scoring': core.SCORING, 'code': {'code/' + name: checksum for name, checksum in core.inventory(code).items()}}
    execution = {'binding': binding, 'bundle': str(bundle), 'detached': True, 'gpu': 'H100',
                 'files': core.inventory(out), 'training': False, 'production_promoted': False}
    core.save(out / 'execution.json', execution)
    checksum = core.sha(out / 'execution.json')
    core.save(out / 'PREPARED.json', {'execution_sha256': checksum})
    core.verify_execution(out, checksum)
    return {'package': str(out.resolve()), 'execution_sha256': checksum, 'cases_per_variant': 2004,
            'variants': core.VARIANTS, 'gpu_launched': False}


def worker_for(package, execution_sha):
    execution, inputs = core.verify_execution(package, execution_sha)
    worker = core.import_file(Path(package) / 'code/worker.py', execution['files']['code/worker.py'])
    return worker, execution, inputs


@contextmanager
def lock_receipt(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + '.lock')
    core.require(not lock.is_symlink() and not path.is_symlink(), 'Linked launch receipt')
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another launcher owns this receipt') from error
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

    def application(self, worker, package, execution_sha):
        return worker.create_app(package, execution_sha, modal_module=self.modal)

    def output(self):
        return self.modal.enable_output()

    def graph(self, call_id):
        return self.modal.FunctionCall.from_id(call_id).get_call_graph()

    def claim(self, worker, run_id, attempt):
        try:
            return self.modal.Dict.from_name(worker.CLAIMS, create_if_missing=False).get(worker.claim_key(run_id, attempt))
        except self.modal.exception.NotFoundError:
            return None

    def remember(self, worker, run_id, attempt, call_id, execution_sha):
        claims = self.modal.Dict.from_name(worker.CLAIMS, create_if_missing=True)
        value = {'call_id': call_id, 'execution_sha256': execution_sha, 'attempt': attempt}
        key = worker.claim_key(run_id, attempt)
        claims.put(key, value, skip_if_exists=True)
        core.require(claims.get(key) == value, 'Another call owns the GPU attempt')

    def progress(self, worker, run_id):
        try:
            volume = self.modal.Volume.from_name(worker.VOLUME, create_if_missing=False)
            raw = b''.join(volume.read_file('/' + worker.PREFIX + '/' + run_id + '/progress.json'))
            value = json.loads(raw)
            core.digest(value)
            return value
        except (FileNotFoundError, self.modal.exception.NotFoundError):
            return None


def reconcile(worker, receipt, backend):
    if not receipt['attempts']:
        return None
    attempt = receipt['attempts'][-1]
    claim = backend.claim(worker, receipt['run_id'], attempt['attempt'])
    if claim:
        core.require(claim['execution_sha256'] == receipt['execution_sha256'] and claim['attempt'] == attempt['attempt']
                     and attempt['call_id'] in (None, claim['call_id']), 'Invocation identity differs')
        attempt.update(call_id=claim['call_id'], spawn_started=True)
    return attempt


def launch(package, execution_sha, receipt_path, *, authorize_gpu=False, resume=False,
           profile='createdliving1000', environment='main', backend=None):
    # Authorization and completed checkpoint validation precede all Modal resource calls.
    core.require(authorize_gpu, 'Explicit --authorize-gpu required; no GPU call made')
    worker, execution, inputs = worker_for(package, execution_sha)
    core.require(core.sha(__file__) == execution['binding']['code']['code/r7_public_benchmark_launch.py'],
                 'Launcher changed since preparation')
    worker.require_launch_python()
    core.bundle_module(execution['bundle'], execution['binding']['bundle_sha256'])
    receipt_path = Path(receipt_path)
    core.require(not receipt_path.resolve().is_relative_to(Path(package).resolve())
                 and not receipt_path.resolve().is_relative_to(Path(execution['bundle']).resolve()),
                 'Receipt must be outside frozen execution and model bundle')
    with lock_receipt(receipt_path):
        if receipt_path.exists():
            core.require(resume, 'Existing receipt; use collect or explicit --resume')
            receipt = core.read(receipt_path)
            core.require(receipt['execution_sha256'] == execution_sha and receipt['run_id'] == execution['binding']['run_id']
                         and receipt['profile'] == profile and receipt['environment'] == environment, 'Receipt identity differs')
        else:
            core.require(not resume, 'No receipt to resume')
            receipt = {'execution_sha256': execution_sha, 'run_id': execution['binding']['run_id'], 'attempts': [],
                       'profile': profile, 'environment': environment, 'state': 'prepared', 'detached': True}
            core.save(receipt_path, receipt)
        backend = backend or ModalBackend(profile, environment)
        previous = reconcile(worker, receipt, backend)
        core.save(receipt_path, receipt)
        attempt_number = 0
        if previous:
            if previous['call_id']:
                worker.terminal_state(backend.graph(previous['call_id']))
                progress = backend.progress(worker, receipt['run_id'])
                if progress is not None:
                    worker.verify_progress(progress, inputs, execution['binding'])
                    if progress['state'] == 'complete':
                        receipt['state'] = 'complete'
                        core.save(receipt_path, receipt)
                        return receipt
                backend.remember(worker, receipt['run_id'], previous['attempt'], previous['call_id'], execution_sha)
                attempt_number = previous['attempt'] + 1
            else:
                core.require(not previous['spawn_started'], 'Submission outcome unknown; recover the claim, do not resubmit')
                attempt_number = previous['attempt']
        else:
            core.require(backend.claim(worker, receipt['run_id'], 0) is None, 'Run ID already has a call; recover its receipt')
        attempt = {'attempt': attempt_number, 'spawn_started': False, 'call_id': None}
        receipt['attempts'].append(attempt)
        receipt['state'] = 'preparing'
        core.save(receipt_path, receipt)
        try:
            app, function = backend.application(worker, package, execution_sha)
            with backend.output(), app.run(detach=True):
                attempt['spawn_started'] = True
                receipt['state'] = 'submitting'
                core.save(receipt_path, receipt)
                call = function.spawn(attempt_number)
                attempt['call_id'] = call.object_id
                receipt.update(state='submitted', app_id=app.app_id)
                core.save(receipt_path, receipt)
                backend.remember(worker, receipt['run_id'], attempt_number, call.object_id, execution_sha)
        except Exception as error:
            receipt['state'] = 'submitted' if attempt['call_id'] else 'submission_unknown' if attempt['spawn_started'] else 'setup_failed'
            attempt['error'] = {'type': type(error).__name__, 'message': str(error)[:2000]}
            core.save(receipt_path, receipt)
            raise
        return receipt


def collect(package, execution_sha, receipt_path, out, *, profile='createdliving1000', environment='main', backend=None):
    worker, execution, inputs = worker_for(package, execution_sha)
    out, receipt_path = Path(out), Path(receipt_path)
    core.require(not out.resolve().is_relative_to(Path(package).resolve())
                 and not out.resolve().is_relative_to(Path(execution['bundle']).resolve()), 'Do not write inside frozen inputs')
    with lock_receipt(receipt_path):
        receipt = core.read(receipt_path)
        core.require(receipt['execution_sha256'] == execution_sha and receipt['run_id'] == execution['binding']['run_id']
                     and receipt['profile'] == profile and receipt['environment'] == environment, 'Collection identity differs')
        backend = backend or ModalBackend(profile, environment)
        reconcile(worker, receipt, backend)
        core.save(receipt_path, receipt)
        progress = backend.progress(worker, receipt['run_id'])
        if progress is None:
            return {'state': 'no_checkpoint_yet', 'cases_per_variant': 2004}
        worker.verify_progress(progress, inputs, execution['binding'])
        latest = out / 'latest.json'
        if latest.exists():
            previous = worker.verify_progress(core.read(latest), inputs, execution['binding'])
            current = {(r['variant'], r['id']): r for r in progress['records']}
            core.require(all(current.get((r['variant'], r['id'])) == r for r in previous['records']), 'Finalized checkpoint regressed')
        snapshot = out / 'snapshots' / (progress['snapshot_sha256'] + '.json')
        if snapshot.exists():
            core.require(core.read(snapshot) == progress, 'Existing snapshot differs')
        else:
            core.save(snapshot, progress)
        row_index = {r['id']: (name, r) for name, rows in inputs.items() for r in rows}
        for record in progress['records']:
            name, row = row_index[record['id']]
            path = out / 'cases' / record['variant'] / name / f'{row["source_index"]:06d}.json'
            value = {'input': row, 'result': record}
            if path.exists():
                core.require(core.read(path) == value, 'Collected case changed')
            else:
                core.save(path, value)
        core.save(latest, progress)
        core.save(out / 'report.json', progress['report'])
        if progress['state'] == 'complete':
            receipt['state'] = 'complete'
            core.save(receipt_path, receipt)
        return {'state': progress['state'], 'total': progress['report']['total'],
                'snapshot_sha256': progress['snapshot_sha256'], 'report': str((out / 'report.json').resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'verify-frozen', 'prepare', 'preflight', 'launch', 'collect'))
    for name in ('results', 'source-launch', 'frozen', 'bundle', 'package', 'receipt', 'out'):
        parser.add_argument('--' + name, type=Path)
    for name in ('frozen-sha', 'bundle-sha', 'execution-sha', 'run-id'):
        parser.add_argument('--' + name)
    parser.add_argument('--profile', default='createdliving1000')
    parser.add_argument('--environment', default='main')
    parser.add_argument('--authorize-gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    required = {'freeze': ('results', 'source_launch', 'out'), 'verify-frozen': ('frozen', 'frozen_sha'),
                'prepare': ('frozen', 'frozen_sha', 'bundle', 'bundle_sha', 'out', 'run_id'),
                'preflight': ('package', 'execution_sha'), 'launch': ('package', 'execution_sha', 'receipt'),
                'collect': ('package', 'execution_sha', 'receipt', 'out')}
    for name in required[args.command]:
        if getattr(args, name) is None:
            parser.error('Missing --' + name.replace('_', '-'))
    if args.command == 'freeze':
        result = core.freeze(args.results, args.source_launch, args.out)
    elif args.command == 'verify-frozen':
        manifest, _ = core.verify_frozen(args.frozen, args.frozen_sha)
        result = {'verified': True, 'counts': manifest['counts'], 'gpu_launched': False}
    elif args.command == 'prepare':
        result = prepare(args.frozen, args.frozen_sha, args.bundle, args.bundle_sha, args.out, args.run_id)
    elif args.command == 'preflight':
        worker, execution, _ = worker_for(args.package, args.execution_sha)
        worker.require_launch_python()
        core.bundle_module(execution['bundle'], execution['binding']['bundle_sha256'])
        result = {'verified': True, 'run_id': execution['binding']['run_id'], 'gpu_launched': False}
    elif args.command == 'launch':
        result = launch(args.package, args.execution_sha, args.receipt, authorize_gpu=args.authorize_gpu,
                        resume=args.resume, profile=args.profile, environment=args.environment)
    else:
        result = collect(args.package, args.execution_sha, args.receipt, args.out, profile=args.profile, environment=args.environment)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
