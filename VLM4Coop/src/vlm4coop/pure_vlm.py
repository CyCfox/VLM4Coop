"""Pure VLM fixed-scenario evaluation with configurable cache use and concurrency."""
import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import time
import torch
from PIL import Image
from vlm4coop.config import load_config, make_args, PROJECT
from vlm4coop.training import JsonLogger, setup_runner, evaluate, make_teacher, append_json
from vlm4coop.response_protocol import none_probabilities


class TeacherMAC:
    def __init__(self, teacher, n_agents, use_cache=None, output=None):
        self.teacher, self.n_agents, self.runner = teacher, n_agents, None
        self.use_cache = (getattr(teacher, 'config', {}).get('pure_vlm_use_cache', True)
                          if use_cache is None else use_cache)
        self.output = Path(output) if output is not None else None
        self.decisions = 0
        self.none_fallbacks = 0
        if self.output is not None:
            (self.output / 'frames').mkdir()

    def init_hidden(self, batch_size):
        pass

    def select_actions(self, ep_batch, t_ep, t_env, bs, test_mode):
        if not test_mode:
            raise ValueError("Pure VLM is evaluation-only")
        # Native runner also selects terminal actions. Never query these states.
        actions = [[0] * self.n_agents for _ in bs]
        positions, images = [], []
        for index, worker in enumerate(bs):
            rgb = self.runner.frames.pop((worker, t_ep), None)
            if rgb is not None:
                positions.append(index)
                images.append(Image.fromarray(rgb))
        if self.output is not None:
            for index, image in zip(positions, images):
                image.save(self.output / 'frames' / f'worker{bs[index]:03d}_t{t_ep:03d}.png')
        began = time.monotonic()
        if hasattr(self.teacher, 'predict_many'):
            results = self.teacher.predict_many(images, self.n_agents, use_cache=self.use_cache)
        else:
            results = [self.teacher.predict(image, self.n_agents, use_cache=self.use_cache) for image in images]
        if len(results) != len(images):
            raise ValueError('Teacher did not return one label per active environment')
        for index, result in zip(positions, results):
            if result.source=='format_failure':
                result=replace(result,probs=none_probabilities(self.n_agents),source='pure_none_fallback')
                self.none_fallbacks+=1
            actions[index] = result.probs.argmax(-1).tolist()
            self.decisions += 1
            if self.output is not None:
                worker = bs[index]
                append_json(self.output / 'pure_vlm_steps.jsonl', dict(worker=worker,
                    env_seed=self.runner.episode_seeds[worker], t=t_ep,
                    actions=actions[index], probabilities=result.probs.tolist(),
                    repaired=result.repaired, cache_hit=result.cache_hit, raw=result.raw,
                    source=result.source,format_attempts=result.format_attempts,decision_id=result.decision_id))
        if self.output is not None and images:
            append_json(self.output / 'batch_timings.jsonl', dict(t=t_ep,
                active_environments=len(images), wall_seconds=time.monotonic()-began,
                cumulative_decisions=self.decisions, cumulative_http_attempts=self.teacher.calls))
        return torch.tensor(actions, dtype=torch.long)



def run(output, config=None, profile='formal'):
    config = deepcopy(config or load_config())
    if profile not in ('formal', 'smoke'):
        raise ValueError(profile)
    if profile == 'formal' and 'pure_vlm' not in config.get('formal_authorized_methods', []):
        raise ValueError('Pure VLM formal evaluation is not authorized')
    config['eval_parallel_envs'] = config.get('pure_vlm_parallel_envs', config['eval_parallel_envs'])
    config['teacher']['concurrency'] = config.get('pure_vlm_concurrency', config['teacher']['concurrency'])
    cache_enabled = config['teacher'].get('pure_vlm_use_cache', True)
    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT.resolve()):
        raise ValueError("Output must be inside VLM4Coop")
    output.mkdir(parents=True, exist_ok=False)
    args = make_args(config, 3, 0)
    args.batch_size_run = config['eval_parallel_envs']
    args.env_args['seed'] = config['fixed_eval_seeds']['start']
    (output / 'config.json').write_text(json.dumps({"experiment_config": config,
        "epymarl_args": vars(args), "method": "pure_vlm", "profile": profile, "agents": 3,
        "action_selection": "per-agent argmax; all-agent NONE after three format failures; no external planner"}, indent=2))
    from vlm4coop.provenance import capture
    capture(output)
    teacher = runner = None
    start = time.monotonic()
    torch.set_num_threads(2)
    (output / 'status.json').write_text(json.dumps(dict(status='running',
        cache_enabled=cache_enabled, parallel_envs=args.batch_size_run,
        concurrency=config['teacher']['concurrency'], started_at_unix=time.time()), indent=2))
    try:
        teacher = make_teacher(config, output, use_cache=cache_enabled,purpose='pure')
        mac = TeacherMAC(teacher, 3, use_cache=cache_enabled, output=output)
        runner, _, _ = setup_runner(args, JsonLogger(output / 'scalars.jsonl'), config, mac)
        runner.capture_frames, mac.runner = True, runner
        summary = evaluate(runner, config, output, scheduled=0, actual=0, seed=None)
        status = dict(status="complete", profile=profile, teacher_calls=teacher.calls,
                      peak_active_http_requests=getattr(teacher, 'max_active_requests', None),
                      cache_enabled=cache_enabled,
                      cache_hits=teacher.cache.hits if teacher.cache else 0,
                      cache_misses=teacher.cache.misses if teacher.cache else 0,
                      cache_writes=teacher.cache.writes if teacher.cache else 0,
                      decisions=mac.decisions, parallel_envs=args.batch_size_run,
                      none_fallback_decisions=mac.none_fallbacks,
                      format_failures=getattr(teacher,'format_failures',0),
                      format_retries=getattr(teacher,'format_retries',0),
                      wall_seconds=time.monotonic() - start, n_eval=summary['n_eval'])
        (output / 'status.json').write_text(json.dumps(status, indent=2))
        return status
    except BaseException as exc:
        (output / 'status.json').write_text(json.dumps(dict(status="failed", error=repr(exc),
            wall_seconds=time.monotonic()-start,
            teacher_calls=teacher.calls if teacher else 0,
            cache_enabled=cache_enabled), indent=2))
        if runner is not None:
            (output / 'partial_episodes.json').write_text(json.dumps(
                getattr(runner, 'last_episode_records', []), indent=2))
        raise
    finally:
        if runner:
            runner.close_env()
        if teacher and teacher.cache:
            teacher.cache.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    print(json.dumps(run(parser.parse_args().output)))
