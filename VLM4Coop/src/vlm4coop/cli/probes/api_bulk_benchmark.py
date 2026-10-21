"""One full ten-environment rollout, followed by one cache-first API batch.

This measures offline annotation throughput, not Pure VLM task success or PPO.
Run only through docker_api.sh; credentials are read by OpenCodeTeacher.
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from vlm4coop.teachers.api import build_api_cache_identity
from vlm4coop.training.configuration import PROJECT, load_config, make_args
from vlm4coop.envs.scenario_sampling import TrainingSeedPool
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.utils.json_logging import JsonLogger
from vlm4coop.utils.json_logging import append_json
from vlm4coop.teachers.factory import make_teacher
from vlm4coop.training.setup import setup_runner


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def label_batch(teacher, images, positions, episodes, output):
    """Submit exactly once, preserving a timing record even on API failure."""
    keys = [teacher.cache.key(image, 3) for image in images]
    unique_keys = set(keys)
    timing = dict(status='running', batch_calls=1, input_states=len(images),
        unique_states=len(unique_keys), duplicate_states=len(images) - len(unique_keys),
        configured_concurrency=teacher.config['concurrency'], top_p=teacher.config['top_p'],
        timer_definition='One predict_many call: cache lookup, deduplication, all HTTP attempts/retries, parsing and SQLite commits; excludes rollout and PNG saving.',
        started_at_unix=time.time())
    save_json(output / 'timing.json', timing)
    started = time.perf_counter()
    try:
        results = teacher.predict_many(images, 3, use_cache=True)
    except Exception as exc:
        timing.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        timing['status'] = 'annotation_complete'
    finally:
        timing.update(batch_wall_seconds=time.perf_counter() - started,
            finished_at_unix=time.time(), http_attempts=teacher.calls,
            peak_active_http_requests=teacher.max_active_requests,
            unique_cache_hits=teacher.cache.hits, unique_cache_misses=teacher.cache.misses,
            cache_writes=teacher.cache.writes, cache_capped=teacher.cache.capped)
        save_json(output / 'timing.json', timing)
    assert len(results) == len(images)
    for (worker, t), key, result in zip(positions, keys, results):
        append_json(output / 'teacher_targets.jsonl', dict(worker=worker, t=t,
            env_seed=episodes[worker]['env_seed'], key=key, probabilities=result.probs.tolist(),
            repaired=result.repaired, cache_hit=result.cache_hit))
    # A new SQLite connection proves durable labels without issuing another API batch.
    verification = ProbabilityCache(teacher.cache.path, build_api_cache_identity(teacher.config),
                                    max_bytes=teacher.cache.max_bytes)
    try:
        for key, result in zip(keys, results):
            cached = verification.get(key)
            assert cached is not None, 'Accepted label was not persisted'
            np.testing.assert_array_equal(cached.probs, result.probs)
    finally:
        verification.close()
    timing.update(status='complete', persistent_cache_verification='passed', verified_states=len(results),
                  verified_unique_states=len(unique_keys))
    save_json(output / 'timing.json', timing)
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-only', action='store_true', help='Docker smoke test without any API request')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / 'results').resolve()):
        raise ValueError('Output must be under VLM4Coop/results')
    output.mkdir(parents=True, exist_ok=False)
    config = load_config()
    assert config['batch_size_run'] == 10
    assert config['environment']['field_size'] == [8, 8]
    assert config['environment']['max_episode_steps'] == 100
    assert config['teacher']['concurrency'] == 1000
    assert config['teacher']['top_p'] == .95
    save_json(output / 'config.json', dict(experiment_config=config,
        purpose='Single-batch offline teacher annotation benchmark; no learning and no Pure VLM control.',
        rollout_policy='Untrained native MAPPO Actor with stochastic actions, seed 0, numeric observations only.',
        agents=3, n_food=3, episodes=10, maximum_joint_steps=1000))
    save_json(output / 'cache_identity.json', build_api_cache_identity(config['teacher']))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    runner = teacher = None
    began = time.perf_counter()
    save_json(output / 'status.json', dict(status='sampling', started_at_unix=time.time()))
    try:
        runner, _, _ = setup_runner(make_args(config, 3, 0, 'smoke'),
                                   JsonLogger(output / 'scalars.jsonl'), config)
        runner.capture_frames = True
        pool = TrainingSeedPool(config)
        seeds = pool.sample()
        save_json(output / 'training_seed_pool.json', dict(pool.metadata, selected_seeds=seeds))
        runner.next_episode_seeds = seeds
        sampling_started = time.perf_counter()
        with torch.no_grad():
            batch = runner.run(test_mode=False)
        sampling_seconds = time.perf_counter() - sampling_started
        episodes = runner.last_episode_records
        assert len(episodes) == 10
        assert [row['env_seed'] for row in episodes] == seeds
        assert len(runner.frames) == runner.t_env == sum(row['length'] for row in episodes)
        assert 0 < runner.t_env <= 1000
        for row in episodes:
            append_json(output / 'episodes.jsonl', row)
        positions = list(runner.frames)
        images = [Image.fromarray(runner.frames[pos]) for pos in positions]
        frame_dir = output / 'frames'
        frame_dir.mkdir()
        for (worker, t), image in zip(positions, images):
            image.save(frame_dir / f'worker{worker:02d}_t{t:03d}.png')
        np.savez_compressed(output / 'numeric_trajectory.npz',
            **{name: batch[name].cpu().numpy() for name in
               ('obs', 'actions', 'avail_actions', 'reward', 'terminated', 'filled')})
        runner.frames.clear()
        runner.close_env()
        runner = None
        status = dict(status='sampled' if args.sample_only else 'annotating', episodes=10,
            joint_steps=len(images), sampling_seconds=sampling_seconds,
            environment_seeds=seeds, http_attempts=0)
        save_json(output / 'status.json', status)
        if not args.sample_only:
            teacher = make_teacher(config, output)
            timing = label_batch(teacher, images, positions, episodes, output)
            status.update(status='complete', http_attempts=timing['http_attempts'],
                          batch_wall_seconds=timing['batch_wall_seconds'])
        status['total_wall_seconds'] = time.perf_counter() - began
        save_json(output / 'status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as exc:
        save_json(output / 'status.json', dict(status='failed', error_type=type(exc).__name__,
            error=str(exc), total_wall_seconds=time.perf_counter() - began))
        raise
    finally:
        if runner is not None:
            runner.close_env()
        if teacher is not None:
            teacher.cache.close()


if __name__ == '__main__':
    main()
