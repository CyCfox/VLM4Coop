"""Fixed initial-scenario pool: independent RNG, actual resets, and audit trace."""
from copy import deepcopy
import json
import numpy as np
import pytest
from vlm4coop.config import load_config, make_args
from vlm4coop.scenario_sampling import TrainingSeedPool, audit_seed_pool
from vlm4coop.environment import register_lbf, CoopGymmaWrapper
from vlm4coop.training import setup_runner, train


def test_pool_draws_are_reproducible_unique_and_rng_isolated():
    config = load_config()
    a, b = TrainingSeedPool(config), TrainingSeedPool(config)
    np.random.seed(75)
    before = np.random.get_state()
    observed = set()
    for _ in range(100):
        x, y = a.sample(), b.sample()
        assert x == y and len(x) == len(set(x)) == 10
        assert all(0 <= seed < 500 for seed in x)
        observed.update(x)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert len(observed) > 400
    # Training RNG is unaffected by arbitrary evaluation RNG changes.
    c, d = TrainingSeedPool(config), TrainingSeedPool(config)
    assert c.sample() == d.sample()
    from vlm4coop.training import evaluation_rng
    with evaluation_rng(1000000): np.random.rand(100)
    assert c.sample() == d.sample()


def test_pool_rejects_eval_overlap_and_too_small_batch():
    config = load_config()
    config['training_seed_pool']['start'] = 1000000
    with pytest.raises(ValueError, match='overlap'): TrainingSeedPool(config)
    config = load_config()
    config['training_seed_pool']['count'] = 9
    with pytest.raises(ValueError, match='smaller'): TrainingSeedPool(config)


def test_all_500_real_scenarios_valid_and_repeated_reset_is_identical():
    config = load_config()
    # Validate the full pool for all three already-planned agent/food counts.
    for n in (3, 4, 5):
        key = register_lbf(n, config['environment'])
        env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None,
                              seed=0, common_reward=True, reward_scalarisation='sum')
        signatures = set()
        try:
            first = {}
            for seed in range(500):
                obs, _ = env.reset(seed=seed)
                assert env.initial_food_count == n
                raw = env._env.unwrapped
                signature = (raw.field.tobytes(), tuple((p.position, p.level) for p in raw.players))
                signatures.add(signature)
                if seed in (0, 17, 499): first[seed] = np.asarray(obs).copy()
            assert len(signatures) == 500
            for seed, expected in first.items():
                env.step([0] * n)
                actual, _ = env.reset(seed=seed)
                assert np.array_equal(expected, np.asarray(actual))
        finally:
            env.close()


def test_actual_parallel_workers_receive_selected_seeds():
    class Log:
        def log_stat(self, *args): pass
    config = load_config()
    args = make_args(config, 3, 0, 'smoke')
    runner, _, _ = setup_runner(args, Log(), config)
    try:
        seeds = TrainingSeedPool(config).sample()
        runner.next_episode_seeds = seeds
        runner.reset()
        observations = runner.batch['obs'][:, 0].numpy()
        key = register_lbf(3, config['environment'])
        env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None,
                              seed=0, common_reward=True, reward_scalarisation='sum')
        try:
            for index, seed in enumerate(seeds):
                obs, _ = env.reset(seed=seed)
                assert np.array_equal(observations[index], np.asarray(obs))
        finally:
            env.close()
        assert runner.episode_seeds == seeds
    finally:
        runner.close_env()
