"""Three cache-first RGB-only LBF closed loops, at most three concurrent HTTP requests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from vlm4coop.teachers.api import api_system_prompt, api_joint_prompt, build_api_cache_identity
from vlm4coop.training.configuration import PROJECT, load_config
from vlm4coop.envs.lbf.adapter import CoopGymmaWrapper, register_lbf
from vlm4coop.utils.provenance import capture
from vlm4coop.teachers.factory import make_teacher
from vlm4coop.utils.json_logging import append_json


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def state(env):
    raw = env._env.unwrapped
    return dict(agents=[dict(id=i, row=int(p.position[0]), col=int(p.position[1]), level=int(p.level))
                        for i, p in enumerate(raw.players)],
                food=[dict(id=env.food_ids[(int(r), int(c))], row=int(r), col=int(c), level=int(raw.field[r,c]))
                      for r, c in np.argwhere(raw.field > 0)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample-only', action='store_true', help='Real resets and rendering, no API or teacher')
    a = p.parse_args()
    output = a.output.resolve()
    if not output.is_relative_to((PROJECT / 'results').resolve()):
        raise ValueError('Output must remain under VLM4Coop/results')
    output.mkdir(parents=True, exist_ok=False)
    config = deepcopy(load_config())
    formal_concurrency = config['teacher']['concurrency']
    config['teacher']['concurrency'] = 3
    assert config['teacher']['thinking'] == {'type': 'disabled'}
    assert 'reasoning_effort' not in config['teacher']
    assert config['environment']['field_size'] == [8,8]
    assert config['environment']['max_episode_steps'] == 100
    seeds = [1200,1201,1202]
    dump(output / 'config.json', dict(experiment_config=config, formal=False, agents=3,
        environment_seeds=seeds, formal_concurrency=formal_concurrency,
        input_mode='single_global_rgb_only', provided_current_coordinates=False,
        action_selection='Per-agent argmax of teacher probabilities, no external planner or action correction',
        system_prompt=api_system_prompt(3), user_prompt=api_joint_prompt(3)))
    dump(output / 'cache_identity.json', build_api_cache_identity(config['teacher']))
    capture(output)
    (output / 'probe_script.py').write_text(Path(__file__).read_text())
    frames = output / 'frames'
    frames.mkdir()
    envs, teacher = [], None
    episodes = [dict(seed=seed, length=0, success=False, team_return=0., remaining_food=3,
                     finished=False) for seed in seeds]
    report = dict(status='sampling', formal=False, configured_concurrency=3, episodes=episodes)
    started = time.monotonic()
    dump(output / 'summary.json', report)
    try:
        key = register_lbf(3, config['environment'])
        for seed in seeds:
            env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None, seed=seed,
                                  common_reward=True, reward_scalarisation='sum')
            envs.append(env)
            env.reset(seed=seed)
            assert sorted(env.food_ids.values()) == [0,1,2]
            Image.fromarray(env.render()).save(frames / f'episode{seed}_initial.png')
            dump(output / f'episode{seed}_initial.json', state(env))
        if a.sample_only:
            report['status'] = 'sampled_no_api'
        else:
            teacher = make_teacher(config, output)
            report['status'] = 'running'
            dump(output / 'summary.json', report)
            for t in range(100):
                active = [i for i, row in enumerate(episodes) if not row['finished']]
                if not active:
                    break
                images = [Image.fromarray(envs[i].render()) for i in active]
                for i, image in zip(active, images):
                    image.save(frames / f'episode{seeds[i]}_step{t:03d}.png')
                before = [state(envs[i]) for i in active]
                began_batch = time.monotonic()
                results = teacher.predict_many(images, 3, use_cache=True)
                append_json(output / 'batch_timings.jsonl', dict(t=t, active_environments=len(active),
                    wall_seconds=time.monotonic()-began_batch, cumulative_http_attempts=teacher.calls))
                for i, image, previous, result in zip(active, images, before, results):
                    actions = result.probs.argmax(-1).tolist()
                    _, reward, terminated, truncated, info = envs[i].step(actions)
                    total_reward = float(np.asarray(reward).sum())
                    row = episodes[i]
                    row.update(length=t+1, team_return=row['team_return']+total_reward,
                        remaining_food=int(info['remaining_food']), success=bool(info['success']),
                        finished=bool(terminated or truncated))
                    append_json(output / 'steps.jsonl', dict(seed=seeds[i], t=t, before=previous,
                        actions=actions, probabilities=result.probs.tolist(), repaired=result.repaired,
                        cache_hit=result.cache_hit, key=teacher.cache.key(image, 3),
                        response_with_visible_reasoning=result.raw, reward=total_reward,
                        after=state(envs[i]), info=info))
                dump(output / 'summary.json', report)
            assert all(row['finished'] for row in episodes)
            report.update(status='complete', successful_episodes=sum(row['success'] for row in episodes),
                success_rate=sum(row['success'] for row in episodes)/3)
        for i, env in enumerate(envs):
            Image.fromarray(env.render()).save(frames / f'episode{seeds[i]}_final.png')
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        if teacher is not None:
            report.update(http_attempts=teacher.calls, peak_active_http_requests=teacher.max_active_requests,
                cache_hits=teacher.cache.hits, cache_misses=teacher.cache.misses,
                cache_writes=teacher.cache.writes, cache_capped=teacher.cache.capped)
            label_path = output / 'teacher_labels.jsonl'
            labels = [json.loads(row) for row in label_path.read_text().splitlines()] if label_path.exists() else []
            report.update(accepted_labels=len(labels), verified_non_thinking_labels=sum(row['reasoning_tokens']==0 for row in labels))
            teacher.cache.close()
        else:
            report['http_attempts'] = 0
        for env in envs:
            env.close()
        dump(output / 'summary.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
