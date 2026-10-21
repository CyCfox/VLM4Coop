"""Verify raw smoke/formal evaluation accounting without inventing outcomes."""
from pathlib import Path
import argparse
import json
from collections import Counter, defaultdict
from vlm4coop.evaluation.metrics import summarize
from vlm4coop.envs.scenario_sampling import audit_seed_pool

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
by_step = defaultdict(list)
for record in records:
    by_step[record['scheduled_eval_step']].append(record)
if config['method'] != 'pure_vlm':
    interval = config['experiment_config']['eval_interval_steps']
    budgets = dict(formal='formal_train_steps',smoke='smoke_train_steps',pilot='pilot_train_steps')
    budget = config['experiment_config'][budgets[config['profile']]]
    expected_steps = list(range(interval, budget+1, interval))
    assert [r['scheduled_eval_step'] for r in evaluations] == expected_steps
    assert sorted(counts) == expected_steps
for result in evaluations:
    selected = by_step[result['scheduled_eval_step']]
    assert len(selected) == expected_n
    seeds = [record['eval_seed'] for record in selected]
    start = config['experiment_config']['fixed_eval_seeds']['start']
    assert sorted(seeds) == list(range(start, start + expected_n))
    horizon=config['experiment_config']['environment']['max_episode_steps']
    assert all(record['length'] <= horizon for record in selected)
    if config['experiment_config'].get('environment_adapter','lbf')=='mpe_simple_spread':
        assert result['scheduled_eval_step'] == result['actual_train_env_steps']
        assert all(record['length']==horizon and record['episode_limit'] for record in selected)
        assert all(record['success']==(record['final_covered_landmarks']==config['agents']) for record in selected)
        assert all(record['environment']=='mpe_simple_spread' and
                   record['final_coverage_fraction']==record['final_covered_landmarks']/config['agents'] for record in selected)
    else:
        assert all(record['success'] == (record['remaining_food'] == 0) for record in selected)
    assert all(record['actual_train_env_steps'] == result['actual_train_env_steps'] for record in selected)
    recomputed = summarize(selected)
    for key in ('success_rate_mean', 'return_mean', 'episodes_length_mean'):
        assert abs(recomputed[key] - result[key]) < 1e-8
pool_validation = audit_seed_pool(run) if config['method'] != 'pure_vlm' else None
print(json.dumps({'status': 'passed', 'training_seed_pool': pool_validation, 'evaluations': len(evaluations),
                  'raw_episodes': len(records), 'profile': config['profile'],
                  'actual_train_env_steps': 0 if config['method']=='pure_vlm' else status['train_env_steps']}))
