"""Native MAPPO orchestration; environment details live in registered adapters."""
import argparse
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from learners.ppo_learner import PPOLearner
from vlm4coop.training.configuration import load_config, make_args
from vlm4coop.utils.paths import PROJECT, ROOT
from vlm4coop.utils.json_logging import append_json, JsonLogger
from vlm4coop.algorithms.distillation import kd_weight
from vlm4coop.algorithms.mappo import CoopPPOLearner
from vlm4coop.training.setup import setup_runner
from vlm4coop.training.annotation import annotate
from vlm4coop.training.checkpoints import _snapshot
from vlm4coop.evaluation.evaluator import evaluate
from vlm4coop.teachers.factory import make_teacher
from vlm4coop.teachers.status import teacher_status
from vlm4coop.envs.scenario_sampling import TrainingSeedPool

def train(method, agents, seed, profile, output_dir, config=None, teacher=None):
    config = deepcopy(config or load_config())
    if method not in {"mappo", "soft", "hard"}:
        raise ValueError(method)
    if profile == "formal" and method not in config.get("formal_authorized_methods", []):
        raise ValueError("Unresolved formal protocol for method " + method)
    output = Path(output_dir).resolve()
    if not output.is_relative_to(PROJECT.resolve()):
        raise ValueError("Output must be inside VLM4Coop")
    output.mkdir(parents=True, exist_ok=False)  # never overwrite a previous result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    args = make_args(config, agents, seed, profile)
    seed_pool = TrainingSeedPool(config) if 'training_seed_pool' in config else None
    if seed_pool is not None:
        (output / 'training_seed_pool.json').write_text(json.dumps(seed_pool.metadata, indent=2))
    logger = JsonLogger(output / "scalars.jsonl")
    train_runner = eval_runner = None
    owned_teacher = False
    protocol_path=(PROJECT/config['protocol_document'] if 'protocol_document' in config
                   else ROOT/'VLM4Coop_Experiment_Plan.md')
    resolved = {"experiment_config": config, "epymarl_args": vars(args),
                "method": method, "agents": agents, "seed": seed, "profile": profile,
                "protocol_source_sha256": sha256(protocol_path.read_bytes()).hexdigest()}
    (output / "config.json").write_text(json.dumps(resolved, indent=2))
    from vlm4coop.utils.provenance import capture
    capture(output)
    start = time.monotonic()
    last_optimized_env_steps = 0
    episode = 0
    (output / 'status.json').write_text(json.dumps(dict(status='running',
        train_env_steps=0, episodes=0, phase='initializing',
        teacher_probability_cache_requested=method!='mappo'), indent=2))
    try:
        train_runner, mac, scheme = setup_runner(args, logger, config)
        eval_args = deepcopy(args)
        eval_args.batch_size_run = config["eval_parallel_envs"]
        eval_args.env_args["seed"] = config["fixed_eval_seeds"]["start"]
        eval_runner, _, _ = setup_runner(eval_args, logger, config, mac)
        args.kd_mode = method if method != "mappo" else "soft"
        args.kd_lambda_current = 0.0
        learner = PPOLearner(mac, scheme, logger, args) if method == "mappo" else CoopPPOLearner(mac, scheme, logger, args)
        if method != "mappo" and teacher is None:
            teacher, owned_teacher = make_teacher(config, output), True
        scheduled, episode, best, saved_steps = config["eval_interval_steps"], 0, None, []
        while train_runner.t_env < args.t_max:
            batch_start = train_runner.t_env
            if seed_pool is not None:
                train_runner.next_episode_seeds = seed_pool.sample()
            train_runner.capture_frames = method != "mappo" and train_runner.t_env < config["kd_anneal_t2"]
            with torch.no_grad():
                batch = train_runner.run(test_mode=False)
            actual = train_runner.t_env
            if seed_pool is not None:
                append_json(output / 'train_batches.jsonl', dict(batch_index=episode // args.batch_size_run,
                    env_seeds=train_runner.next_episode_seeds,
                    batch_start_env_steps=batch_start, actual_train_env_steps=actual))
            args.kd_lambda_current = 0 if method == "mappo" else kd_weight(
                actual, config["kd_lambda_initial"], config["kd_lambda_plateau"],
                config["kd_anneal_t1"], config["kd_anneal_t2"])
            for item in train_runner.last_episode_records:
                append_json(output / "train_episodes.jsonl", dict(item, actual_train_env_steps=actual))
            if args.kd_lambda_current > 0:
                annotate(train_runner, batch, teacher)
            else:
                train_runner.frames.clear()
            # Match epymarl/src/run.py: discard trailing padding before reward
            # standardisation and native n-step return computation.
            batch = batch[:, :int(batch.max_t_filled())]
            learner.train(batch, actual, episode)
            last_optimized_env_steps = actual
            episode += args.batch_size_run
            while scheduled <= actual and scheduled <= args.t_max:
                summary = evaluate(eval_runner, config, output, scheduled, actual, seed)
                best = _snapshot(learner, output, actual, summary, best, config["retain_checkpoints"], saved_steps)
                scheduled += config["eval_interval_steps"]
            status = {"status": "running", "train_env_steps": actual,
                      "episodes": episode, "wall_seconds": time.monotonic() - start}
            status.update(teacher_status(teacher))
            (output / "status.json").write_text(json.dumps(status, indent=2))
        status.update(status="complete", teacher_calls=teacher.calls if teacher else 0)
        (output / "status.json").write_text(json.dumps(status, indent=2))
        return status
    except BaseException as exc:
        failed = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
        failed.update(status='failed', error=repr(exc),
                      sampled_env_steps=train_runner.t_env if train_runner is not None else 0,
                      train_env_steps=last_optimized_env_steps, episodes=episode,
                      phase='failed', wall_seconds=time.monotonic()-start)
        failed.update(teacher_status(teacher))
        (output / "status.json").write_text(json.dumps(failed, indent=2))
        raise
    finally:
        for runner in (train_runner, eval_runner):
            if runner is not None:
                runner.close_env()
        if owned_teacher:
            teacher.cache.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mappo", "soft", "hard"], required=True)
    parser.add_argument("--agents", type=int, choices=[3, 4, 5], default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=["smoke", "pilot", "formal"], default="smoke")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="Frozen experiment YAML; defaults to the existing LBF protocol")
    args = parser.parse_args()
    print(json.dumps(train(args.method, args.agents, args.seed, args.profile, args.output,
                           config=load_config(args.config) if args.config else None)))


if __name__ == "__main__":
    main()
