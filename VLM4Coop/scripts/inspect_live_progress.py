"""Read local progress evidence only: no API calls, no policy-score inspection."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import statistics
import time

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
a = parser.parse_args()
def read(name):
    path = a.run / name
    rows=[]
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass  # concurrently written last line
    return rows
train=read('train_episodes.jsonl'); evaluations=read('eval_episodes.jsonl')
targets=read('teacher_targets.jsonl'); labels=read('teacher_labels.jsonl')
calls=read('teacher_calls.jsonl'); scalars=read('scalars.jsonl'); retries=read('transport_retries.jsonl')
latest=max((x['actual_train_env_steps'] for x in train),default=0)
expected=sum(x['length'] for x in train if x['actual_train_env_steps']==latest)
latest_targets=[x for x in targets if x['actual_train_env_steps']==latest]
optimized=sorted({x['step'] for x in scalars if x['metric']=='kd_loss'})
clock=datetime.now(timezone(timedelta(hours=8)))
def mtime(name):
    p=a.run/name
    return datetime.fromtimestamp(p.stat().st_mtime,timezone(timedelta(hours=8))).isoformat() if p.exists() else None
recent=[x for x in calls if x.get('completed_at_unix',0)>time.time()-300]
lat=[x['latency_seconds'] for x in calls]
result=dict(observed_at=clock.isoformat(), run=str(a.run), train_episode_rows=len(train),
    eval_episode_rows=len(evaluations), last_sampled_env_steps=latest,
    ppo_updates_completed=len(optimized),last_optimized_env_steps=optimized[-1] if optimized else 0,
    latest_batch_required_labels=expected,latest_batch_labels_committed=len(latest_targets),
    latest_batch_cache_hits=sum(x['cache_hit'] for x in latest_targets),
    all_target_rows=len(targets),accepted_new_labels=len(labels),completed_api_attempts=len(calls),
    http_status_counts=dict(Counter(str(x.get('http_status')) for x in calls)),
    retry_events=len(retries),attempts_finished_last_5_minutes=len(recent),
    http_status_last_5_minutes=dict(Counter(str(x.get('http_status')) for x in recent)),
    mean_api_latency_seconds=statistics.mean(lat) if lat else None,
    max_api_latency_seconds=max(lat) if lat else None,
    generated_tokens=sum(x.get('response',{}).get('usage',{}).get('completion_tokens',0) for x in calls),
    reasoning_tokens=sum(x.get('response',{}).get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0) for x in calls),
    responses_after_last_target_flush=sum(x.get('completed_at_unix',0)>(a.run/'teacher_targets.jsonl').stat().st_mtime for x in calls) if (a.run/'teacher_targets.jsonl').exists() else 0,
    mtimes={name:mtime(name) for name in ('train_episodes.jsonl','eval_episodes.jsonl','teacher_calls.jsonl','teacher_labels.jsonl','teacher_targets.jsonl','scalars.jsonl')})
path=Path('VLM4Coop/reports/go_live_progress_'+clock.strftime('%Y%m%d_%H%M%S')+'.json')
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
