"""Restore only real, validated responses that a failed batch never cached."""
import argparse
import json
from pathlib import Path
import numpy as np
from vlm4coop.api_teacher import api_joint_prompt, api_system_prompt, build_api_cache_identity, off_response_evidence
from vlm4coop.config import PROJECT
from vlm4coop.rendering import RENDERER_VERSION
from vlm4coop.teacher import ProbabilityCache, TeacherResult, parse_probabilities
from vlm4coop.response_protocol import parse_api_probabilities

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
a = parser.parse_args()
frozen = json.loads((a.run / 'config.json').read_text())
config, n = frozen['experiment_config'], frozen['agents']
identity = build_api_cache_identity(config['teacher'])
# Recovery uses each audit label's already-derived key, never rekeys a legacy
# response under different top_p or other new generation parameters.
responses = {}
for line in (a.run / 'teacher_calls.jsonl').read_text().splitlines():
    event = json.loads(line)
    if not event.get('valid_transport_completion'): continue
    assert event['endpoint'] == config['teacher']['endpoint']
    assert event['model_requested'] == config['teacher']['model']
    assert event['system'] == api_system_prompt(n) and event['prompt'] == api_joint_prompt(n)
    assert event['thinking'] == config['teacher']['thinking']
    response = event['response']
    assert response.get('id') is not None and response['id'] not in responses
    responses[response['id']] = event
cache = ProbabilityCache(PROJECT / 'cache/teacher.sqlite', identity, config['cache_max_bytes'])
restored = existing = 0
try:
    for line in (a.run / 'teacher_labels.jsonl').read_text().splitlines():
        label = json.loads(line)
        if label['key'] is None:
            continue  # uncached Pure VLM labels must never be inserted into SQLite
        event = responses[label['response_id']]
        message = event['response']['choices'][0]['message']
        raw = message['content']
        expected_effort = config['teacher'].get('reasoning_effort', 'provider_default')
        if config['teacher']['thinking']['type'] == 'disabled':
            if label['control_attempt'] == 1 and label.get('control_mode') != 'native_off_retry':
                expected_effort = 'provider_default' if expected_effort == 'none' else 'none'
            evidence = off_response_evidence(event['response'])
            assert evidence != 'contradicted'
            assert label['reasoning_tokens'] == event['response'].get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')
            if 'thinking_evidence' in label:
                assert label['thinking_evidence'] == evidence
            else:
                assert evidence == 'reported_zero'  # preserve legacy evidence requirements
        assert event['reasoning_effort'] == expected_effort
        probs,repaired=(parse_api_probabilities(raw,n,label['purpose']) if 'purpose' in label
                        else parse_probabilities(raw,n))
        assert np.allclose(probs, label['probabilities']) and repaired == label['repaired']
        if cache.get(label['key']) is not None:
            existing += 1
        else:
            assert cache.put(label['key'], TeacherResult(probs, raw, repaired))
            restored += 1
finally:
    cache.close()
result = dict(status='passed', source=str(a.run), validated_real_labels=len(responses),
              already_cached=existing, restored=restored, new_api_requests=0,
              generated_or_modified_probabilities=0)
(PROJECT / 'reports/go_cache_recovery_20261017.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
