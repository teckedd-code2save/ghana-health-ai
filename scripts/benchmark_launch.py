#!/usr/bin/env python3
"""Create one frozen benchmark pack and launch detached matched evaluations."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MODAL_PROFILE', 'createdliving1000')
os.environ.setdefault('MODAL_ENVIRONMENT', 'main')
ROOT = Path(__file__).resolve().parents[1]


def terminal_state(graph):
    """Unknown/disconnected/pending calls cannot authorize another GPU job."""
    if not graph:
        raise ValueError('No terminal call evidence; do not launch a duplicate')
    states = []
    for node in graph:
        state = node.status.name
        if state not in {'SUCCESS', 'FAILURE', 'INIT_FAILURE', 'TERMINATED', 'TIMEOUT'}:
            raise ValueError('Prior call is still pending; do not launch a duplicate')
        states.append(state)
        if node.children:
            states.append(terminal_state(node.children))
    return 'SUCCESS' if all(s == 'SUCCESS' for s in states) else 'FAILED'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--variants', nargs='+', choices=['base', 'r5', 'r6'], default=['base', 'r5', 'r6'])
    args = parser.parse_args()
    if args.receipt.exists() and not args.resume:
        parser.error('Receipt already exists; explicitly resume rather than duplicate jobs')
    import modal
    spec = importlib.util.spec_from_file_location('benchmark_r6', ROOT / 'modal/train/benchmark_r6.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with modal.enable_output(), module.app.run(detach=True):
        if args.resume:
            receipt = json.loads(args.receipt.read_text())
            if receipt['run_id'] != args.run_id:
                parser.error('Resume identity differs')
        else:
            receipt = module.prepare.remote(args.run_id, args.limit)
        receipt.setdefault('calls', [])
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        for variant in args.variants:
            previous = [c for c in receipt['calls'] if c['variant'] == variant]
            states = []
            for call in previous:
                graph = modal.FunctionCall.from_id(call['call_id']).get_call_graph()
                states.append(terminal_state(graph))
            if states and states[-1] == 'SUCCESS':
                print(f'{variant} already completed; not resubmitted', flush=True)
                continue
            call = module.benchmark.spawn(args.run_id, variant, receipt['manifest_sha256'])
            receipt['calls'].append({'variant': variant, 'call_id': call.object_id})
            args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
            print(f'Submitted {variant}: {call.object_id}', flush=True)
    print('Detached jobs running. Receipt:', args.receipt, flush=True)


if __name__ == '__main__':
    main()
