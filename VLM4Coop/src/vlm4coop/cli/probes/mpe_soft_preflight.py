"""Sparse real-API MPE integration check; never a formal experiment score."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch

from learners.ppo_learner import PPOLearner
from vlm4coop.algorithms.mappo import CoopPPOLearner
from vlm4coop.envs.scenario_sampling import TrainingSeedPool
from vlm4coop.evaluation.evaluator import evaluate
from vlm4coop.teachers.factory import make_teacher
from vlm4coop.teachers.status import teacher_status
from vlm4coop.training.annotation import annotate
from vlm4coop.training.configuration import load_config, make_args
from vlm4coop.training.setup import setup_runner
from vlm4coop.utils.json_logging import JsonLogger
from vlm4coop.utils.paths import PROJECT
from vlm4coop.utils.provenance import capture
from vlm4coop.utils.rgb_frames import frame_image


def run(output, labels=8):
    if not 3 <= labels <= 16: raise ValueError('Preflight selects3..16states only')
    output = output.resolve()
    if not output.is_relative_to(PROJECT.resolve()): raise ValueError('Project-owned output required')
    output.mkdir(exist_ok=False)
    c = load_config(PROJECT/'configs/mpe_soft1000.yaml')
    (output/'config.json').write_text(json.dumps(dict(experiment_config=c,
        formal=False, profile='engineering_sparse_labels', selected_states=labels), indent=2))
    capture(output)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(2)
    args = make_args(c, 3, 0, 'smoke'); args.kd_mode = 'soft'; args.kd_lambda_current = .1
    logger = JsonLogger(output/'scalars.jsonl')
    runner = eval_runner = teacher = None; started = time.monotonic()
    try:
        runner, mac, scheme = setup_runner(args, logger, c)
        runner.next_episode_seeds = TrainingSeedPool(c).sample(); runner.capture_frames = True
        with torch.no_grad(): batch = runner.run()
        assert runner.t_env == 1000 and len(runner.last_episode_records) == 40
        assert len(runner.frames) == 1000
        teacher = make_teacher(c, output)
        chosen = []; seen = set()
        for item in runner.frames.items():
            key = teacher.cache.key(frame_image(item[1]), 3)
            if key in seen: continue
            seen.add(key); chosen.append(item)
            if len(chosen) == labels: break
        assert len(chosen) == labels
        for (worker, t), frame in chosen:
            frame_image(frame).save(output/f'worker{worker:03d}_t{t:03d}.png', format='PNG')
        runner.frames = dict(chosen); reward = batch['reward'].clone()
        annotate(runner, batch, teacher)
        assert int(batch['teacher_valid'].sum()) == labels and torch.equal(batch['reward'], reward)
        http_after_annotation = teacher.calls
        for _, frame in chosen:
            assert teacher.predict(frame_image(frame), 3).cache_hit
        assert teacher.calls == http_after_annotation
        batch = batch[:, :int(batch.max_t_filled())]
        baseline = PPOLearner(deepcopy(mac), scheme, logger, deepcopy(args))
        learner = CoopPPOLearner(deepcopy(mac), scheme, logger, deepcopy(args))
        learner.critic.load_state_dict(baseline.critic.state_dict())
        learner.target_critic.load_state_dict(baseline.target_critic.state_dict())
        baseline.train(batch, 1000, 0); learner.train(batch, 1000, 0)
        difference = max((a-b).abs().max().item() for a,b in zip(learner.mac.parameters(), baseline.mac.parameters()))
        assert difference > 1e-8 and args.epochs == 4
        eval_args = deepcopy(args); eval_args.batch_size_run = c['eval_parallel_envs']
        eval_runner, _, _ = setup_runner(eval_args, logger, c, learner.mac)
        evaluation = evaluate(eval_runner, c, output, 1000, 1000, 0)
        assert evaluation['n_eval'] == 100 and teacher.calls == http_after_annotation
        result = dict(status='passed', formal=False, full_training_episodes=40,
            sampled_steps=1000, selected_teacher_states=labels, native_PPO_epochs=4,
            cache_persisted_and_reused=True, reward_unchanged=True,
            actor_difference_vs_zero_KD=difference, student_evaluation_episodes=100,
            API_requests_during_PPO_and_eval=0, configured_HTTP_concurrency=1000,
            actual_HTTP_concurrency_claim='See measured peak;8state smoke is not1000HTTP proof',
            wall_seconds=time.monotonic()-started, **teacher_status(teacher))
        (output/'status.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result
    except BaseException as exc:
        (output/'status.json').write_text(json.dumps(dict(status='failed', formal=False,
            error=repr(exc), wall_seconds=time.monotonic()-started, **teacher_status(teacher)), indent=2))
        raise
    finally:
        for r in (runner, eval_runner):
            if r is not None: r.close_env()
        if teacher is not None: teacher.cache.close()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--labels', type=int, default=8); a=p.parse_args(); run(a.output, a.labels)


if __name__ == '__main__': main()
