"""Transport -> real-image perception -> closed-loop capability, never formal data."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import yaml
from vlm4coop.teachers.api import OpenCodeTeacher, ApiRequestError
from vlm4coop.training.configuration import PROJECT, load_config
from vlm4coop.envs.lbf.adapter import register_lbf, CoopGymmaWrapper


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def state(env):
    raw = env._env.unwrapped
    return {'agents': [{'id': i, 'row': int(p.position[0]), 'col': int(p.position[1]),
                         'level': int(p.level)} for i, p in enumerate(raw.players)],
            'food': [{'row': int(r), 'col': int(c), 'level': int(raw.field[r, c])}
                     for r, c in zip(*np.nonzero(raw.field))]}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    config = yaml.safe_load((PROJECT / 'configs/api_probe.yaml').read_text())
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = PROJECT / 'results' / ('api_teacher_probe_' + stamp)
    output.mkdir(exist_ok=False)
    dump(output / 'config.json', dict(config, episodes=args.episodes,
         environment=load_config()['environment'], input_mode='rgb_only', provided_coordinates=False,
         note='Exploratory API capability test; not part of formal experiments'))
    from vlm4coop.utils.provenance import capture
    capture(output)
    print(json.dumps({'output': str(output)}), flush=True)
    teacher = OpenCodeTeacher(config, output / 'requests.jsonl')
    phase, env = 'connectivity', None
    report = {'profile': 'api_capability_probe', 'formal': False, 'model': config['model'],
              'input_mode': 'rgb_only', 'provided_coordinates': False,
              'thinking': config.get('thinking'), 'reasoning_effort': config.get('reasoning_effort', 'provider_default')}
    dump(output / 'summary.json', report)
    try:
        answer, response = teacher.request('Reply with exactly API_OK.', label='connectivity')
        report['connectivity'] = {'success': 'API_OK' in answer,
                                  'reply': answer, 'returned_model': response.get('model')}
        dump(output / 'summary.json', report)
        print(json.dumps({'phase': phase, **report['connectivity']}), flush=True)
        phase = 'real_image_perception'
        env_config = load_config()['environment']
        key = register_lbf(3, env_config)
        env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None, seed=0,
                              common_reward=True, reward_scalarisation='sum')
        env.reset(seed=1234)
        image = Image.fromarray(env.render())
        image.save(output / 'perception_input.png')
        ground_truth = state(env)
        dump(output / 'perception_ground_truth.json', ground_truth)
        prompt = ('Read only the attached grid image. Blue circles are agents A0,A1,A2; '
                  'green squares F are food. L denotes level. Rows are the labels at the left '
                  'and increase downward; columns are labels at the top and increase rightward. '
                  'Return only JSON {"agents":[{"id":0,"row":0,"col":0,"level":1},...],'
                  '"food":[{"row":0,"col":0,"level":1},...]}. '
                  'Report all three agents and all visible food with their actual coordinates '
                  'and levels. Do not guess if you cannot access the image.')
        answer, response = teacher.request(prompt, image, label=phase)
        (output / 'perception_answer.txt').write_text(answer)
        try:
            start = answer.index('{')
            perceived, _ = json.JSONDecoder().raw_decode(answer[start:])
            score = {name: sorted(perceived[name], key=lambda x: json.dumps(x, sort_keys=True)) ==
                           sorted(ground_truth[name], key=lambda x: json.dumps(x, sort_keys=True))
                     for name in ('agents', 'food')}
        except (ValueError, KeyError, TypeError):
            score = {'agents': False, 'food': False, 'parse_failed': True}
        report['perception'] = score
        dump(output / 'summary.json', report)
        print(json.dumps({'phase': phase, **score}), flush=True)
        if not score['agents'] or not score['food']:
            report.update(status='image_perception_failed', calls=teacher.calls)
            dump(output / 'summary.json', report)
            return 1
        phase = 'closed_loop'
        records = []
        for seed in range(1200, 1200 + args.episodes):
            env.reset(seed=seed)
            total_reward, successes = 0., False
            for t in range(100):
                image = Image.fromarray(env.render())
                image.save(output / f'episode_{seed}_step_{t:03d}.png')
                before = state(env)
                result = teacher.predict(image, 3, use_cache=False)
                actions = result.probs.argmax(-1).tolist()
                _, reward, terminated, truncated, info = env.step(actions)
                reward_sum = float(np.asarray(reward).sum())
                total_reward += reward_sum
                step = dict(seed=seed, t=t, before=before, actions=actions,
                            probabilities=result.probs.tolist(), repaired=result.repaired,
                            reward=reward_sum, after=state(env), info=info)
                with (output / 'steps.jsonl').open('a') as stream:
                    stream.write(json.dumps(step) + '\n')
                dump(output / 'progress.json', dict(seed=seed, step=t+1,
                     remaining_food=int(info['remaining_food']), api_calls=teacher.calls))
                if terminated or truncated:
                    successes = bool(info['success'])
                    break
            record = dict(seed=seed, length=t+1, success=successes, team_return=total_reward,
                          remaining_food=int(info['remaining_food']))
            records.append(record)
            report['episodes'] = records
            dump(output / 'summary.json', report)
            print(json.dumps({'phase': phase, **record}), flush=True)
        report.update(status='complete', calls=teacher.calls)
    except (ApiRequestError, ValueError) as exc:
        report.update(status='failed', failed_phase=phase, error=str(exc), calls=teacher.calls)
        print(json.dumps({'status': 'failed', 'phase': phase, 'error': str(exc)}), flush=True)
        return 1
    finally:
        if env is not None:
            env.close()
        dump(output / 'summary.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    raise SystemExit(run())
