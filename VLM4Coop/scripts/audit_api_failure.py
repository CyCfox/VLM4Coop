"""Read-only summary of a stopped experiment; never prints full API responses."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
a = parser.parse_args()
def rows(name):
    p = a.run / name
    if p.exists():
        with p.open() as stream:
            for line in stream:
                if line.strip(): yield json.loads(line)
calls = list(rows('teacher_calls.jsonl'))
labels = list(rows('teacher_labels.jsonl'))
train = list(rows('train_episodes.jsonl'))
scalars = list(rows('scalars.jsonl'))
evals = list(rows('evaluations.jsonl'))
keys = {row['key'] for row in labels}
cache = sqlite3.connect('file:VLM4Coop/cache/teacher.sqlite?mode=ro', uri=True)
cached = sum(cache.execute('SELECT 1 FROM predictions WHERE key=?', (key,)).fetchone() is not None for key in keys)
cache.close()
errors = [{k: row.get(k) for k in ('call_index', 'http_status', 'error_body', 'error', 'latency_seconds')}
          for row in calls if row.get('http_status') != 200 or row.get('error')]
p = a.run / 'teacher_calls.jsonl'
result = dict(http_status_counts=dict(Counter(str(row.get('http_status')) for row in calls)),
    total_requests=len(calls), accepted_labels=len(labels), unique_accepted_keys=len(keys),
    accepted_keys_already_cached=cached, accepted_keys_not_cached=len(keys)-cached,
    last_sampled_env_steps=max((x['actual_train_env_steps'] for x in train), default=0),
    last_ppo_update_env_steps=max((x['step'] for x in scalars if x['metric']=='kd_loss'), default=0),
    completed_evaluation_points=len(evals),
    last_audit_write=datetime.fromtimestamp(p.stat().st_mtime,timezone(timedelta(hours=8))).isoformat(),
    errors=errors, billing_balance_queried=False)
output = a.run / 'failure_audit.json'
output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
