"""Summarize actual exploratory API requests; never promote them to formal results."""
import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
root = args.run
summary = json.loads((root / 'summary.json').read_text())
requests = [json.loads(x) for x in (root / 'requests.jsonl').read_text().splitlines()]
rows = []
for event in requests:
    response = event.get('response', {})
    usage = response.get('usage', {})
    choices = response.get('choices', [])
    rows.append({'label': event['label'], 'http_status': event.get('http_status'),
                 'finish_reason': choices[0].get('finish_reason') if choices else None,
                 'latency_seconds': event['latency_seconds'],
                 'prompt_tokens': usage.get('prompt_tokens', 0),
                 'completion_tokens': usage.get('completion_tokens', 0),
                 'reasoning_tokens': usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0),
                 'has_final_answer': bool(event.get('valid_transport_completion'))})
policy = [row for row in rows if row['label'].startswith('policy_')]
episodes = summary.get('episodes', [])
result = {'profile': 'exploratory_api_capability', 'formal': False,
          'run_status': summary.get('status', 'running_or_unfinished'),
          'requests': rows, 'total_requests_returned': len(rows),
          'policy_requests_returned': len(policy),
          'policy_final_answers': sum(row['has_final_answer'] for row in policy),
          'policy_latency_mean_seconds': statistics.mean(row['latency_seconds'] for row in policy) if policy else None,
          'prompt_tokens_total': sum(row['prompt_tokens'] for row in rows),
          'completion_tokens_total': sum(row['completion_tokens'] for row in rows),
          'reasoning_tokens_total': sum(row['reasoning_tokens'] for row in rows),
          'episodes_completed': len(episodes),
          'successes': sum(row['success'] for row in episodes),
          'note': 'Unfinished episodes are never counted as failed or successful. No training took place.'}
(root / 'request_statistics.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
