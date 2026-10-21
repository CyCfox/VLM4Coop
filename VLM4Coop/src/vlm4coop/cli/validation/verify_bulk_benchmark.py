"""Offline cache/response audit after a single-batch API benchmark, no API key."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from vlm4coop.training.configuration import PROJECT
from vlm4coop.teachers.api import build_api_cache_identity
from vlm4coop.teachers.cache import ProbabilityCache, parse_probabilities

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
a = p.parse_args()
run = a.run
config = json.loads((run / 'config.json').read_text())['experiment_config']
timing = json.loads((run / 'timing.json').read_text())
events = [json.loads(row) for row in (run / 'teacher_calls.jsonl').read_text().splitlines()]
labels = [json.loads(row) for row in (run / 'teacher_labels.jsonl').read_text().splitlines()]
cache = ProbabilityCache(PROJECT / 'cache/teacher.sqlite', build_api_cache_identity(config['teacher']), config['cache_max_bytes'])
try:
    responses = {row['response']['id']: row for row in events if row.get('valid_transport_completion')}
    for label in labels:
        row = responses[label['response_id']]
        assert row['top_p_requested'] == .95
        assert row['max_tokens'] is None
        parsed, repaired = parse_probabilities(row['response']['choices'][0]['message']['content'], 3)
        cached = cache.get(label['key'])
        assert cached is not None
        np.testing.assert_array_equal(cached.probs, parsed)
        np.testing.assert_array_equal(cached.probs, np.asarray(label['probabilities'], dtype=np.float32))
        assert repaired == label['repaired']
    positions = []
    for frame in (run / 'frames').glob('worker*_t*.png'):
        worker, t = frame.stem.split('_t')
        positions.append((int(worker.removeprefix('worker')), int(t), frame))
    targets = []
    missing = set()
    keys = set()
    for worker, t, frame in sorted(positions):
        with Image.open(frame) as image:
            key = cache.key(image, 3)
        keys.add(key)
        cached = cache.get(key)
        record = dict(worker=worker, t=t, key=key, available=cached is not None)
        if cached is not None:
            record.update(probabilities=cached.probs.tolist(), repaired=cached.repaired)
        else:
            record.update(probabilities=None, error='No accepted probability label; see original HTTP audit.')
            missing.add(key)
        targets.append(record)
finally:
    cache.close()
assert len(targets) == timing['input_states'] and len(keys) == timing['unique_states']
assert len(labels) == timing['cache_writes']
with (run / 'audited_targets.jsonl').open('x') as stream:
    for row in targets:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
failed = []
for row in events:
    if row.get('http_status') == 200 and not row.get('valid_transport_completion'):
        response = row.get('response', {})
        message = (response.get('choices') or [{}])[0].get('message', {})
        failed.append(dict(call_index=row['call_index'], response_id=response.get('id'),
            finish_reason=(response.get('choices') or [{}])[0].get('finish_reason'),
            usage=response.get('usage'), model_returned=response.get('model'),
            final_text_length=len(message.get('content') or ''),
            reasoning_text_length=len(message.get('reasoning_content') or ''),
            latency_seconds=row['latency_seconds']))
summary = dict(status='partial' if missing else 'complete', offline_audit='passed',
    api_calls_during_audit=0, input_states=len(targets), unique_states=len(keys),
    accepted_unique_labels=len(labels), verified_cache_labels=len(labels),
    available_input_states=sum(row['available'] for row in targets),
    unavailable_input_states=sum(not row['available'] for row in targets),
    missing_unique_states=len(missing), missing_keys=sorted(missing),
    http_attempts=len(events), http_status_counts=dict(Counter(str(row.get('http_status', 'transport_error')) for row in events)),
    repaired_accepted_labels=sum(row['repaired'] for row in labels),
    all_requests_top_p_095=all(row['top_p_requested'] == .95 for row in events),
    all_requests_thinking_enabled_low=all(row['thinking']=={'type':'enabled'} and row['reasoning_effort']=='low' for row in events),
    failed_completions=failed, original_batch_wall_seconds=timing['batch_wall_seconds'])
(run / 'offline_audit.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
print(json.dumps(summary, indent=2, allow_nan=False))
