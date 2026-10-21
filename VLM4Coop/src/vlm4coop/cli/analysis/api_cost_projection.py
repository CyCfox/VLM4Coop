"""Explicit no-cache cost/time extrapolation, not a bill or a guaranteed budget."""
import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
summary = json.loads((args.run / 'summary.json').read_text())
count = sum(row['length'] for row in summary.get('episodes', []))
requests = [json.loads(x) for x in (args.run / 'requests.jsonl').read_text().splitlines()]
policy = [row for row in requests if row['label'].startswith('policy_')][:count]
if not policy:
    raise ValueError('No completed-episode decision data')
mean_input = statistics.mean(row['response']['usage']['prompt_tokens'] for row in policy)
mean_output = statistics.mean(row['response']['usage']['completion_tokens'] for row in policy)
mean_latency = statistics.mean(row['latency_seconds'] for row in policy)
rates = {'input_usd_per_million': .30, 'output_usd_per_million': 1.20}
states = 400000
cost = states * (mean_input * rates['input_usd_per_million'] +
                 mean_output * rates['output_usd_per_million']) / 1e6
result = {'price_source': 'https://opencode.ai/v2/docs/console/models/',
          'price_checked_date': '2026-10-17', 'rates': rates,
          'reference_completed_episodes': len(summary.get('episodes', [])),
          'reference_policy_requests': len(policy), 'reference_agents': 3,
          'mean_input_tokens': mean_input, 'mean_output_tokens': mean_output,
          'mean_latency_seconds': mean_latency, 'example_supervised_states': states,
          'example_no_cache_cost_usd': cost,
          'example_serial_days': states * mean_latency / 86400,
          'actual_account_charge_known': False,
          'limitations': ['Extrapolation only, neither a bill nor an upper bound.',
            'Assumes a new request for every state; cache hits reduce requests.',
            'Assumes observed 3-agent tokens/latency; 4/5 agents are not yet measured.',
            'Account plan, discounts, prompt caching, retries and Pure VLM not included.',
            'Concurrent requests can reduce elapsed time, subject to provider limits.']}
(args.run / 'cost_projection.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
