"""Native EPyMARL MAPPO orchestration with isolated fixed-scenario evaluation."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import math
import random
import shutil
import time
import numpy as np
import torch
import yaml
from components.transforms import OneHot
from controllers.basic_controller import BasicMAC
from learners.ppo_learner import PPOLearner
from envs import REGISTRY
from vlm4coop.config import PROJECT, ROOT, load_config, make_args
from vlm4coop.distillation import kd_weight
from vlm4coop.environment import make_env, register_lbf
from vlm4coop.learner import CoopPPOLearner
from vlm4coop.metrics import checkpoint_rank, summarize
from vlm4coop.runner import CoopParallelRunner
from vlm4coop.teacher import ProbabilityCache, LocalQwenTeacher, joint_prompt, SYSTEM_PROMPT
from vlm4coop.rendering import RENDERER_VERSION
from vlm4coop.scenario_sampling import TrainingSeedPool
from PIL import Image


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None  # JSON null denotes a missing completion statistic (NaN in CSV).
    return value


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(_json_safe(value), allow_nan=False) + "\n")


class JsonLogger:
    def __init__(self, path):
        self.path = path

    def log_stat(self, key, value, t):
        append_json(self.path, {"step": int(t), "metric": key, "value": float(value)})


@contextmanager
def evaluation_rng(seed):
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def setup_runner(args, logger, config, mac=None):
    register_lbf(args.n_agents, config["environment"])
    REGISTRY["vlm4coop"] = make_env
    runner = CoopParallelRunner(args, logger)
    info = runner.get_env_info()
    assert info["obs_shape"] == 6 * args.n_agents, info
    args.n_actions = info["n_actions"]
    scheme = {"state": {"vshape": info["state_shape"]},
              "obs": {"vshape": info["obs_shape"], "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)},
              "terminated": {"vshape": (1,), "dtype": torch.uint8},
              "teacher_probs": {"vshape": (6,), "group": "agents"},
              "teacher_valid": {"vshape": (1,)}}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    # EpisodeBatch preprocessing adds the one-hot scheme before MAC construction.
    from components.episode_buffer import EpisodeBatch
    template = EpisodeBatch(scheme, groups, 1, info["episode_limit"] + 1, preprocess=preprocess)
    mac = mac or BasicMAC(template.scheme, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    return runner, mac, template.scheme


def annotate(runner, batch, teacher):
    """Freeze pre-action targets once; allow the configured HTTP concurrency."""
    try:
        items = list(runner.frames.items())
        chunk_size = max(32, getattr(teacher, 'config', {}).get('concurrency', 1))
        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            if any(t >= batch.max_seq_length - 1 for (worker, t), rgb in chunk):
                raise ValueError("Captured terminal/padding frame")
            images = [Image.fromarray(rgb) for _, rgb in chunk]
            if hasattr(teacher, 'predict_many'):
                results = teacher.predict_many(images, runner.args.n_agents)
            else:
                results = [teacher.predict(image, runner.args.n_agents) for image in images]
            if len(results) != len(chunk):
                raise ValueError("Teacher returned a different number of state labels")
            resolved=[]
            for ((worker, t), _), result in zip(chunk, results):
                if result.source=='format_failure':
                    # Storage placeholder only, not a teacher distribution.
                    # PPO retains its native rollout mask; only KD is disabled.
                    result=replace(result,probs=np.zeros((runner.args.n_agents,6),dtype=np.float32),source='kd_masked')
                resolved.append(result)
                batch.update({"teacher_probs": torch.from_numpy(result.probs)[None, None],
                              "teacher_valid": torch.full((1,1,1),float(result.source!='kd_masked'))},
                             bs=[worker], ts=t, mark_filled=False)
            if hasattr(teacher, 'audit_path'):
                with teacher.audit_path.with_name('teacher_targets.jsonl').open('a') as stream:
                    for ((worker, t), _), image, result in zip(chunk, images, resolved):
                        record = dict(actual_train_env_steps=runner.t_env, worker=worker, t=t,
                            key=teacher.cache.key(image, runner.args.n_agents) if teacher.cache else None,
                            probabilities=result.probs.tolist() if result.source!='kd_masked' else None,
                            teacher_valid=int(result.source!='kd_masked'),cache_hit=result.cache_hit,
                            repaired=result.repaired,source=result.source,
                            format_attempts=result.format_attempts,decision_id=result.decision_id)
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
    finally:
        runner.frames.clear()


def evaluate(runner, config, output_dir, scheduled, actual, seed):
    start = config["fixed_eval_seeds"]["start"]
    count = config["eval_episodes_every_checkpoint"]
    if count % runner.batch_size:
        raise ValueError("Evaluation must contain full parallel batches")
    records = []
    # Same scenarios AND action RNG at every checkpoint. Restore training RNG.
    with evaluation_rng(start):
        for base in range(0, count, runner.batch_size):
            runner.next_episode_seeds = list(range(start + base, start + base + runner.batch_size))
            with torch.no_grad():
                runner.run(test_mode=True)
            for index, episode in enumerate(runner.last_episode_records):
                record = dict(episode, eval_seed=start + base + index, train_seed=seed,
                              scheduled_eval_step=scheduled, actual_train_env_steps=actual)
                append_json(output_dir / "eval_episodes.jsonl", record)
                records.append(record)
    summary = summarize(records)
    summary.update(scheduled_eval_step=scheduled, actual_train_env_steps=actual, train_seed=seed)
    append_json(output_dir / "evaluations.jsonl", summary)
    return summary


def make_teacher(config, output, teacher_type=LocalQwenTeacher, use_cache=True, purpose='distillation'):
    if config["teacher"].get("backend") == "openai_api":
        from vlm4coop.api_teacher import OpenCodeTeacher, build_api_cache_identity
        identity = build_api_cache_identity(config['teacher'])
        cache = (ProbabilityCache(PROJECT / "cache/teacher.sqlite", identity,
                                  max_bytes=config["cache_max_bytes"]) if use_cache else None)
        return OpenCodeTeacher(config["teacher"], output / "teacher_calls.jsonl", cache,purpose=purpose)
    model_path = Path(config["teacher"]["model_path"])
    identity = dict(config["teacher"], renderer=RENDERER_VERSION, system_prompt=SYSTEM_PROMPT,
                    model_config=json.loads((model_path / "config.json").read_text()),
                    weights=[(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                             for p in sorted(model_path.glob("*.safetensors"))],
                    prompt_templates={n: joint_prompt(n) for n in (3, 4, 5)})
    cache = (ProbabilityCache(PROJECT / "cache/teacher.sqlite", identity,
                              max_bytes=config["cache_max_bytes"]) if use_cache else None)
    return teacher_type(config["teacher"], cache, output / "teacher_calls.jsonl")


def _snapshot(learner, output, actual, summary, best, retain, manifest):
    checkpoint_dir = output / "checkpoints" / str(actual)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
        learner.save_models(str(checkpoint_dir))
        # Native save_models lacks target network/running moments; preserve them.
        extra = {"target_critic": learner.target_critic.state_dict(),
                 "old_mac": learner.old_mac.agent.state_dict(),
                 "critic_training_steps": learner.critic_training_steps}
        for name in ("rew_ms", "ret_ms"):
            if hasattr(learner, name):
                obj = getattr(learner, name)
                extra[name] = {k: getattr(obj, k) for k in ("mean", "var", "count")}
        torch.save(extra, checkpoint_dir / "learner_extra.th")
    manifest.append(actual)
    current_rank = checkpoint_rank(summary)
    if best is None or current_rank > best:
        best = current_rank
        # The best pointer keeps its directory from retention cleanup.
        (output / "best_checkpoint.json").write_text(json.dumps(_json_safe({"train_env_steps": actual,
                                                                 "rank": list(best)}), allow_nan=False))
    best_step = json.loads((output / "best_checkpoint.json").read_text())["train_env_steps"]
    keep = set(manifest[-retain:]) | {best_step}
    for path in (output / "checkpoints").iterdir():
        if path.is_dir() and int(path.name) not in keep:
            shutil.rmtree(path)  # only this run's explicitly-owned checkpoints
    return best


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
    resolved = {"experiment_config": config, "epymarl_args": vars(args),
                "method": method, "agents": agents, "seed": seed, "profile": profile,
                "protocol_source_sha256": sha256((ROOT / "VLM4Coop_Experiment_Plan.md").read_bytes()).hexdigest()}
    (output / "config.json").write_text(json.dumps(resolved, indent=2))
    from vlm4coop.provenance import capture
    capture(output)
    start = time.monotonic()
    last_optimized_env_steps = 0
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
            if teacher is not None and teacher.cache is not None:
                status.update(teacher_calls=teacher.calls, cache_hits=teacher.cache.hits,
                              cache_misses=teacher.cache.misses, cache_writes=teacher.cache.writes,
                              teacher_probability_cache_enabled=True,
                              peak_active_http_requests=getattr(teacher,'max_active_requests',None),
                              format_failures=getattr(teacher,'format_failures',0),
                              format_retries=getattr(teacher,'format_retries',0),
                              thinking_control_retries=getattr(teacher,'thinking_control_retries',0),
                              rejected_thinking_responses=getattr(teacher,'thinking_control_failures',0),
                              teacher_format_failed_decisions=getattr(teacher,'failed_decisions',0))
            (output / "status.json").write_text(json.dumps(status, indent=2))
        status.update(status="complete", teacher_calls=teacher.calls if teacher else 0)
        (output / "status.json").write_text(json.dumps(status, indent=2))
        return status
    except BaseException as exc:
        failed = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
        failed.update(status='failed', error=repr(exc),
                      sampled_env_steps=train_runner.t_env if train_runner is not None else 0,
                      train_env_steps=last_optimized_env_steps)
        if teacher is not None:
            failed['teacher_calls'] = teacher.calls
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
    args = parser.parse_args()
    print(json.dumps(train(args.method, args.agents, args.seed, args.profile, args.output)))


if __name__ == "__main__":
    main()
