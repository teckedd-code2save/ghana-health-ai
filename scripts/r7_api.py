"""Protected VPS relay with durable requests, bounded spending and no blind retries."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

# Standard short-context USD per million tokens; verified 2026-11-17.
# https://developers.openai.com/api/docs/models/gpt-6.2-sol
RATES = {'gpt-6.2-sol': (2, 10), 'gpt-5.6-sol': (4, 20), 'gpt-6.2-astra': (8, 40)}


def blob(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(blob(value)).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        stream.write(blob(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_bytes())


def transport(body):
    js = """(async()=>{let b='';for await(const c of process.stdin)b+=c;
    const k=process.env.OPENAI_API_KEY;if(!k)throw Error('missing credential');
    const r=await fetch('https://api.openai.com/v1/responses',{method:'POST',redirect:'error',
    headers:{Authorization:'Bearer '+k,'Content-Type':'application/json'},body:b,
    signal:AbortSignal.timeout(240000)});
    console.log(JSON.stringify({status:r.status,request_id:r.headers.get('x-request-id'),
    body:await r.json()}))})().catch(()=>process.exit(2));"""
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-i',
               os.getenv('SSH_KEY', str(Path.home()/'.ssh/optimi_github_actions')),
               'root@128.140.12.62', 'docker exec -i ghana-health-ai-web node -e ' + shlex.quote(js)]
    result = subprocess.run(command, input=blob(body), capture_output=True, timeout=260, check=True)
    return json.loads(result.stdout)


def response_text(raw):
    if raw.get('status') != 'completed':
        raise ValueError('Incomplete provider response; retained for diagnosis')
    chunks = [part['text'] for item in raw.get('output', []) if item.get('type') == 'message'
              for part in item.get('content', []) if part.get('type') == 'output_text']
    if len(chunks) != 1:
        raise ValueError('Expected one structured response')
    return json.loads(chunks[0])


class Ledger:
    def __init__(self, root, cap_usd=50):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cap_micro = int(cap_usd * 1_000_000)
        if self.cap_micro <= 0:
            raise ValueError('Positive finite spending allowance required')

    @contextmanager
    def lock(self):
        with (self.root/'.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def summary(self):
        receipts = [read(p) for p in (self.root/'receipts').glob('*.json')]
        states = {}
        for item in receipts:
            states[item['state']] = states.get(item['state'], 0) + 1
        return {'requests': len(receipts), 'states': states,
                'estimated_usd': sum(r.get('estimated_micro', 0) for r in receipts)/1_000_000,
                'reserved_or_spent_usd': sum(r['charge_bound_micro'] for r in receipts)/1_000_000,
                'cap_usd': self.cap_micro/1_000_000,
                'billing_status': 'token-based estimate, not account balance'}

    def call(self, body, *, sender=transport):
        if body.get('model') not in RATES or body.get('store') is not False or body.get('service_tier') != 'default':
            raise ValueError('Unpriced model or unexpected retention/service tier')
        if body.get('tools') or body.get('tool_choice') != 'none':
            raise ValueError('External tools must run separately with explicit evidence')
        output = body.get('max_output_tokens')
        if type(output) is not int or not 1 <= output <= 8192 or len(blob(body)) > 100_000:
            raise ValueError('Unbounded request')
        signature = digest(body)
        path = self.root/'receipts'/f'{signature}.json'
        rawpath = self.root/'responses'/f'{signature}.json'
        rate_in, rate_out = RATES[body['model']]
        bound = (len(blob(body))+4096)*rate_in + output*rate_out
        with self.lock():
            if path.exists():
                receipt = read(path)
                if receipt['payload_sha256'] != signature:
                    raise ValueError('Request identity changed')
                if receipt['state'] != 'completed':
                    raise ValueError(f"Prior request {receipt['state']}; reconcile, do not resend: {signature}")
                raw = read(rawpath)
                if digest(raw) != receipt['response_sha256']:
                    raise ValueError('Saved response changed')
                return response_text(raw['body']), receipt
            used = sum(read(p)['charge_bound_micro'] for p in (self.root/'receipts').glob('*.json'))
            if used + bound > self.cap_micro:
                raise ValueError('API spending allowance reached')
            receipt = {'state':'pending', 'payload_sha256':signature, 'charge_bound_micro':bound,
                       'model':body['model'], 'started':time.time(), 'rates_per_million':[rate_in,rate_out]}
            save(self.root/'requests'/f'{signature}.json', body)
            save(path, receipt)
        try:
            raw = sender(body)
        except Exception as exc:
            with self.lock():
                save(path, {**receipt, 'state':'unknown', 'error_type':type(exc).__name__})
            raise RuntimeError(f'Relay interrupted; request retained as unknown: {signature}') from None
        with self.lock():
            save(rawpath, raw)
            usage = raw.get('body', {}).get('usage') or {}
            receipt.update(state='http_error', response_sha256=digest(raw), http_status=raw['status'],
                           request_id=raw.get('request_id'), response_id=raw.get('body',{}).get('id'), ended=time.time())
            if all(type(usage.get(k)) is int for k in ('input_tokens','output_tokens')):
                cost = usage['input_tokens']*rate_in + usage['output_tokens']*rate_out
                receipt.update(usage=usage, estimated_micro=cost, charge_bound_micro=(cost*5+3)//4)
            save(path, receipt)
            if raw['status'] != 200:
                raise RuntimeError(f"Provider HTTP {raw['status']}; see protected receipt {signature}")
            try:
                parsed = response_text(raw['body'])
            except ValueError:
                save(path, {**receipt, 'state':'invalid_output'})
                raise
            receipt['state'] = 'completed'
            save(path, receipt)
        return parsed, receipt


def request(system, content, schema, *, name='r7_annotation', model='gpt-6.2-sol', max_tokens=4096):
    return {'model':model, 'store':False, 'service_tier':'default', 'tools':[], 'tool_choice':'none',
            'reasoning':{'effort':'low'}, 'max_output_tokens':max_tokens,
            'input':[{'role':'system','content':system}, {'role':'user','content':json.dumps(content,ensure_ascii=False)}],
            'text':{'format':{'type':'json_schema','name':name,'schema':schema,'strict':True}}}
