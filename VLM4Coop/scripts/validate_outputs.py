"""Verify raw smoke/formal evaluation accounting without inventing outcomes."""
from pathlib import Path
import argparse
import json
from collections import Counter
from vlm4coop.metrics import summarize
from vlm4coop.scenario_sampling import audit_seed_pool

parser = argparse.ArgumentParser()
parser.add_argument('run')
args = parser.parse_args()
run = Path(args.run)
config = json.loads((run / 'config.json').read_text())
status = json.loads((run / 'status.json').read_text())
assert status['status'] == 'complete', status
records = [json.loads(line) for line in (run / 'eval_episodes.jsonl').read_text().splitlines()]
evaluations = [json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]
expected_n = config['experiment_config']['eval_episodes_every_checkpoint']
counts = Counter(record['scheduled_eval_step'] for record in records)
assert set(counts.values()) == {expected_n}, counts
for result in evaluations:
    selected = [record for record in records if record['scheduled_eval_step'] == result['scheduled_eval_step']]
    assert len(selected) == expected_n
    seeds = [record['eval_seed'] for record in selected]
    start = config['experiment_config']['fixed_eval_seeds']['start']
    assert sorted(seeds) == list(range(start, start + expected_n))
    assert all(record['length'] <= 100 for record in selected)
    assert all(record['success'] == (record['remaining_food'] == 0) for record in selected)
    assert all(record['actual_train_env_steps'] == result['actual_train_env_steps'] for record in selected)
    recomputed = summarize(selected)
    for key in ('success_rate_mean', 'return_mean', 'episodes_length_mean'):
        assert abs(recomputed[key] - result[key]) < 1e-8
pool_validation = audit_seed_pool(run) if config['method'] != 'pure_vlm' else None
print(json.dumps({'status': 'passed', 'training_seed_pool': pool_validation, 'evaluations': len(evaluations),
                  'raw_episodes': len(records), 'profile': config['profile'],
                  'actual_train_env_steps': status['train_env_steps']}))
