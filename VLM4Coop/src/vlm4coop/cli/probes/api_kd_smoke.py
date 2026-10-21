"""Real API -> cached labels -> native PPO -> isolated evaluation engineering test.

Nonformal, deliberately labels a selected subset of pre-action states from ten
complete episodes. This is not the full-supervision formal experiment protocol.
"""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
import argparse
import random
import time
import numpy as np
import torch
from PIL import Image
from learners.ppo_learner import PPOLearner
from vlm4coop.training.configuration import PROJECT, load_config, make_args
from vlm4coop.training.setup import setup_runner
from vlm4coop.training.annotation import annotate
from vlm4coop.evaluation.evaluator import evaluate
from vlm4coop.utils.json_logging import JsonLogger
from vlm4coop.teachers.factory import make_teacher
from vlm4coop.algorithms.mappo import CoopPPOLearner
from vlm4coop.utils.provenance import capture
from vlm4coop.envs.scenario_sampling import TrainingSeedPool
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=int, default=8)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    debug_labels = options.labels
    if not 1 <= debug_labels <= 100:
        raise ValueError('Debug labels must be between 1 and 100')
    config = deepcopy(load_config())
    # Respect the latest user-selected teacher mode; never force thinking off.
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = (options.output or PROJECT / 'results' / ('api_kd_debug_' + stamp)).resolve()
    if not output.is_relative_to(PROJECT.resolve()):
        raise ValueError('Output must stay inside VLM4Coop')
    output.mkdir(exist_ok=False)
    (output / 'config.json').write_text(json.dumps(dict(experiment_config=config,
        profile='engineering_debug_sparse_labels', formal=False,
        selected_debug_pre_action_states=debug_labels, episodes_per_batch=10, ppo_epochs=4,
        note='Sparse labels are ONLY for this integration test, never the formal protocol.'), indent=2))
    capture(output)
    print(json.dumps({'output': str(output)}), flush=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(2)
    args = make_args(config, 3, 0, 'smoke')
    args.kd_mode, args.kd_lambda_current = 'soft', .1
    logger = JsonLogger(output / 'scalars.jsonl')
    runner = eval_runner = teacher = None
    started = time.monotonic()
    try:
        runner, mac, scheme = setup_runner(args, logger, config)
        if 'training_seed_pool' in config:
            runner.next_episode_seeds = TrainingSeedPool(config).sample()
        runner.capture_frames = True
        with torch.no_grad():
            batch = runner.run(test_mode=False)
        sampled_states = len(runner.frames)
        assert sampled_states == runner.t_env
        assert len(runner.last_episode_records) == 10
        teacher = make_teacher(config, output)
        selected, seen = [], set()
        for item in runner.frames.items():
            key = teacher.cache.key(Image.fromarray(item[1]), 3)
            if key not in seen:
                selected.append(item)
                seen.add(key)
            if len(selected) == debug_labels:
                break
        assert len(selected) == debug_labels
        for (worker, t), rgb in selected:
            Image.fromarray(rgb).save(output / f'teacher_worker{worker}_t{t}.png')
        runner.frames = dict(selected)
        annotate(runner, batch, teacher)
        valid_states = int(batch['teacher_valid'].sum())
        assert not runner.frames and 0 < valid_states <= debug_labels
        calls_after_annotation = teacher.calls
        first_valid = next(rgb for (worker,t),rgb in selected if batch['teacher_valid'][worker,t].item())
        cached = teacher.predict(Image.fromarray(first_valid), 3)
        assert cached.cache_hit and teacher.calls == calls_after_annotation
        batch = batch[:, :int(batch.max_t_filled())]
        baseline = PPOLearner(deepcopy(mac), scheme, logger, deepcopy(args))
        learners = {}
        for mode in ('soft', 'hard'):
            mode_args = deepcopy(args)
            mode_args.kd_mode = mode
            learner = CoopPPOLearner(deepcopy(mac), scheme, logger, mode_args)
            learner.critic.load_state_dict(baseline.critic.state_dict())
            learner.target_critic.load_state_dict(baseline.target_critic.state_dict())
            learners[mode] = learner
        baseline.train(batch, runner.t_env, 0)
        differences = {}
        for mode, learner in learners.items():
            learner.train(batch, runner.t_env, 0)
            difference = max((a-b).abs().max().item() for a,b in
                             zip(learner.mac.parameters(), baseline.mac.parameters()))
            assert difference > 1e-8, 'Actual KD did not change the trained actor relative to zero-KD'
            differences[mode] = difference
            checkpoint = output / 'checkpoints' / mode
            checkpoint.mkdir(parents=True)
            learner.save_models(str(checkpoint))
        assert teacher.calls == calls_after_annotation
        eval_args = deepcopy(args)
        eval_args.env_args['seed'] = config['fixed_eval_seeds']['start']
        eval_runner, _, _ = setup_runner(eval_args, logger, config, learners['soft'].mac)
        evaluation = evaluate(eval_runner, config, output, 1000, runner.t_env, 0)
        assert evaluation['n_eval'] == 100 and teacher.calls == calls_after_annotation
        label_file = output / 'teacher_labels.jsonl'
        accepted = [json.loads(row) for row in label_file.read_text().splitlines()] if label_file.exists() else []
        assert len(accepted) == teacher.cache.writes and all(row['requested_thinking'] == config['teacher']['thinking'] for row in accepted), 'Accepted labels did not use the configured thinking mode'
        result = dict(status='passed', profile='engineering_debug_sparse_labels', formal=False,
            sampled_joint_states=sampled_states, full_episodes=10, actual_train_steps=runner.t_env,
            sampled_environment_seeds=[row['env_seed'] for row in runner.last_episode_records],
            selected_debug_labels=debug_labels, actual_api_requests=calls_after_annotation,
            valid_teacher_states=valid_states, masked_teacher_states=debug_labels-valid_states,
            configured_concurrency=config['teacher'].get('concurrency', 1),
            max_actual_concurrent_requests=teacher.max_active_requests,
            cache_hits=teacher.cache.hits, ppo_epochs_per_mode=4,
            cache_writes=teacher.cache.writes,
            training_cache_enabled=teacher.cache is not None,
            thinking_control_retries=teacher.thinking_control_retries,
            rejected_thinking_responses=teacher.thinking_control_failures,
            endpoint=config['teacher']['endpoint'],
            max_actor_parameter_difference_vs_zero_kd=differences,
            student_eval_episodes=100, api_requests_during_ppo_and_eval=0,
            wall_seconds=time.monotonic()-started)
        (output / 'status.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        (output / 'status.json').write_text(json.dumps(dict(status='failed', error=str(exc))))
        raise
    finally:
        for item in (runner, eval_runner):
            if item is not None:
                item.close_env()
        if teacher is not None:
            teacher.cache.close()


if __name__ == '__main__':
    main()
