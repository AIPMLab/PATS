"""Sequential job queue for journal-revision experiments.

- Jobs: JSONL, one JSON object per line: {"id", "cmd", "expect_csv"}
- Resumable: a job is skipped when its expect_csv already exists.
- GPU discipline: before each job, wait until free VRAM >= MIN_FREE_MB
  (another training job shares the L40; never launch into low memory).
- Status: protocol/status.json (override with JR_STATUS) (id -> pending|running|done|failed|skipped)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
LOG_DIR = os.path.join(HERE, 'logs')
STATUS = os.environ.get('JR_STATUS', os.path.join(HERE, 'queue_status.json'))
MIN_FREE_MB = 18000
POLL_S = 60

PYTHON = sys.executable  # run inside the same env that launches this script


def gpu_free_mb():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
        return min(int(x) for x in out.decode().split())
    except Exception:
        return 99999


def load_status():
    if os.path.exists(STATUS):
        with open(STATUS) as f:
            return json.load(f)
    return {}


def save_status(st):
    with open(STATUS, 'w') as f:
        json.dump(st, f, indent=1)


def run_job(job, st):
    log_path = os.path.join(LOG_DIR, job['id'] + '.log')
    with open(log_path, 'a') as log:
        log.write(f'\n==== START {time.strftime("%F %T")} ====\n')
        log.write(' '.join(job['cmd']) + '\n')
        log.flush()
        proc = subprocess.run(job['cmd'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    st[job['id']] = 'done' if proc.returncode == 0 else f'failed(rc={proc.returncode})'
    save_status(st)


def main(jobs_file):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(jobs_file) as f:
        jobs = [json.loads(l) for l in f if l.strip() and not l.startswith('#')]
    st = load_status()
    for job in jobs:
        jid = job['id']
        expect = job.get('expect_csv')
        if expect and os.path.exists(os.path.join(ROOT, expect)):
            st[jid] = 'skipped(results exist)'
            save_status(st)
            print(f'[skip] {jid}', flush=True)
            continue
        if str(st.get(jid, '')).startswith('done'):
            print(f'[done] {jid}', flush=True)
            continue
        while gpu_free_mb() < MIN_FREE_MB:
            print(f'[wait] {jid}: free VRAM {gpu_free_mb()}MB < {MIN_FREE_MB}MB; '
                  f'retrying in {POLL_S}s', flush=True)
            time.sleep(POLL_S)
        if expect and os.path.exists(os.path.join(ROOT, expect)):
            st[jid] = 'skipped(results exist)'
            save_status(st)
            print(f'[skip] {jid}', flush=True)
            continue
        print(f'[run ] {jid}: {" ".join(job["cmd"])}', flush=True)
        st[jid] = 'running'
        save_status(st)
        run_job(job, st)
        print(f'[{st[jid]}] {jid}', flush=True)
    print('queue finished', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
