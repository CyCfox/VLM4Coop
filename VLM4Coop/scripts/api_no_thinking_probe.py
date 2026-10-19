"""User-authorized diagnostic: no-thinking RGB vs RGB+coordinates, paired seeds.

Coordinates are privileged numeric inputs in the second condition. They are
explicitly logged and this condition must not be described as RGB-only VLM.
"""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import yaml
from vlm4coop.api_teacher import OpenCodeTeacher, ApiRequestError, api_joint_prompt, api_system_prompt
from vlm4coop.teacher import parse_probabilities
from vlm4coop.config import PROJECT, load_config
from vlm4coop.environment import register_lbf, CoopGymmaWrapper
from vlm4coop.provenance import capture
from api_teacher_probe import state, dump


def run_condition(root, mode, teacher_config, environment):
    output = root / mode
    output.mkdir(exist_ok=False)
    config = dict(teacher_config, thinking={'type': 'disabled'}, reasoning_effort='none', temperature=0)
    resolved = {'teacher': config, 'environment': environment,
                'input_mode': mode, 'coordinates_include_levels': False,
                'seeds': [1200, 1201, 1202], 'max_episode_steps': 100,
                'profile': 'diagnostic_no_thinking', 'formal': False}
    dump(output / 'config.json', resolved)
    capture(output)
    teacher = OpenCodeTeacher(config, output / 'requests.jsonl')
    report = {'status': 'running', 'profile': 'diagnostic_no_thinking', 'formal': False,
              'input_mode': mode, 'model': config['model'], 'episodes': []}
    dump(output / 'summary.json', report)
    key = register_lbf(3, environment)
    env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None, seed=0,
                          common_reward=True, reward_scalarisation='sum')
    started = time.monotonic()
    try:
        for seed in resolved['seeds']:
            env.reset(seed=seed)
            total = 0.
            for t in range(100):
                before = state(env)
                image = Image.fromarray(env.render())
                image.save(output / f'episode_{seed}_step_{t:03d}.png')
                text = api_joint_prompt(3)
                if mode == 'rgb_coordinates':
                    coordinates = {'agents': [{k: row[k] for k in ('id', 'row', 'col')}
                                              for row in before['agents']],
                                   'food': [{k: row[k] for k in ('row', 'col')}
                                            for row in before['food']]}
                    text += ('\nAdditional exact coordinates from the environment (levels are shown in the image): '
                             + json.dumps(coordinates, separators=(',', ':')))
                raw, response = teacher.request(text, image=image, system=api_system_prompt(3),
                                                 label=f'policy_3p_seed{seed}_t{t}')
                message = response['choices'][0]['message']
                reasoning_tokens = response.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')
                emitted_reasoning = bool(message.get('reasoning_content'))
                if emitted_reasoning or reasoning_tokens not in (None, 0):
                    raise ValueError('Provider emitted reasoning despite thinking.disabled; condition invalid')
                probs, repaired = parse_probabilities(raw, 3)
                actions = probs.argmax(-1).tolist()
                _, reward, terminated, truncated, info = env.step(actions)
                reward_sum = float(np.asarray(reward).sum())
                total += reward_sum
                record = dict(seed=seed, t=t, before=before, actions=actions,
                    probabilities=probs.tolist(), repaired=repaired, reward=reward_sum,
                    after=state(env), info=info, reasoning_tokens=reasoning_tokens,
                    emitted_reasoning=emitted_reasoning)
                with (output / 'steps.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                dump(output / 'progress.json', dict(seed=seed, step=t+1,
                    remaining_food=int(info['remaining_food']), calls=teacher.calls,
                    wall_seconds=time.monotonic()-started))
                if terminated or truncated:
                    break
            episode = dict(seed=seed, length=t+1, success=bool(info['success']),
                           team_return=total, remaining_food=int(info['remaining_food']))
            report['episodes'].append(episode)
            dump(output / 'summary.json', report)
            print(json.dumps(dict(mode=mode, **episode)), flush=True)
        report.update(status='complete', calls=teacher.calls,
                      wall_seconds=time.monotonic()-started)
    except (ApiRequestError, ValueError) as exc:
        report.update(status='failed', error=str(exc), calls=teacher.calls,
                      wall_seconds=time.monotonic()-started)
        print(json.dumps({'mode': mode, 'status': 'failed', 'error': str(exc)}), flush=True)
    finally:
        env.close()
        dump(output / 'summary.json', report)
    return report


def main():
    base = yaml.safe_load((PROJECT / 'configs/api_probe.yaml').read_text())
    base.pop('profile', None)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    root = PROJECT / 'results' / ('api_no_thinking_' + stamp)
    root.mkdir(exist_ok=False)
    print(json.dumps({'output': str(root)}), flush=True)
    reports = {}
    for mode in ('rgb_only', 'rgb_coordinates'):
        reports[mode] = run_condition(root, mode, base, load_config()['environment'])
        dump(root / 'summary.json', reports)
    return 0 if all(row['status']=='complete' for row in reports.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
