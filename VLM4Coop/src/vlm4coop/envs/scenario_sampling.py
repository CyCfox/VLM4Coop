"""Reproducible training scenarios, independent of policy/evaluation RNGs."""
import json
from pathlib import Path
import numpy as np


class TrainingSeedPool:
    def __init__(self, config):
        pool = config['training_seed_pool']
        self.batch_size = config['batch_size_run']
        for key in ('start', 'count', 'sampler_seed'):
            if type(pool[key]) is not int or pool[key] < 0:
                raise ValueError('Training pool parameters must be nonnegative integers')
        if pool['count'] < self.batch_size:
            raise ValueError('Training pool is smaller than the rollout batch')
        if pool['sampling'] != 'uniform_without_replacement_per_batch':
            raise ValueError('Unsupported scenario sampling rule')
        self.seeds = list(range(pool['start'], pool['start'] + pool['count']))
        eval_start = config['fixed_eval_seeds']['start']
        evaluation = set(range(eval_start, eval_start + config['eval_episodes_every_checkpoint']))
        if evaluation.intersection(self.seeds):
            raise ValueError('Training and evaluation seed pools overlap')
        self.rng = np.random.Generator(np.random.PCG64(pool['sampler_seed']))
        self.metadata = dict(seeds=self.seeds, sampler_seed=pool['sampler_seed'],
            rng='numpy.Generator(PCG64)', batch_size=self.batch_size,
            replacement_within_batch=False, replacement_between_batches=True)

    def sample(self):
        return [int(seed) for seed in self.rng.choice(self.seeds, self.batch_size, replace=False)]


def audit_seed_pool(run):
    """Replay the recorded draw sequence and check actual worker reset seeds."""
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())['experiment_config']
    if 'training_seed_pool' not in config:
        return None
    sampler = TrainingSeedPool(config)
    metadata = json.loads((run / 'training_seed_pool.json').read_text())
    assert metadata == sampler.metadata
    batches = [json.loads(row) for row in (run / 'train_batches.jsonl').read_text().splitlines()]
    records = [json.loads(row) for row in (run / 'train_episodes.jsonl').read_text().splitlines()]
    assert len(records) == len(batches) * sampler.batch_size
    seen, cumulative = set(), 0
    for index, batch in enumerate(batches):
        expected = sampler.sample()
        assert batch['batch_index'] == index and batch['env_seeds'] == expected
        selected = records[index * sampler.batch_size:(index + 1) * sampler.batch_size]
        assert [row['worker'] for row in selected] == list(range(sampler.batch_size))
        assert [row['env_seed'] for row in selected] == expected
        assert batch['batch_start_env_steps'] == cumulative
        cumulative += sum(row['length'] for row in selected)
        assert batch['actual_train_env_steps'] == cumulative
        assert all(row['actual_train_env_steps'] == cumulative for row in selected)
        seen.update(expected)
    status = json.loads((run / 'status.json').read_text())
    if status['status'] == 'complete':
        assert cumulative == status['train_env_steps']
        assert len(records) == status['episodes']
    for row in (run / 'eval_episodes.jsonl').read_text().splitlines():
        episode = json.loads(row)
        assert episode['env_seed'] == episode['eval_seed']
        assert episode['env_seed'] not in sampler.seeds
    return dict(status='passed', pool_size=len(sampler.seeds), batches=len(batches),
                unique_training_seeds_seen=len(seen),
                sequence_replayed=True, actual_worker_seeds_verified=True,
                train_eval_seed_overlap=False)
