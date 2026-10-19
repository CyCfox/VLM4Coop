"""Replay completed exploratory episodes against actual LBF; never call an API."""
import argparse
import json
from pathlib import Path
import tarfile
import yaml
import numpy as np
from PIL import Image
from vlm4coop.environment import register_lbf, CoopGymmaWrapper
from api_teacher_probe import state

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
root = args.run
summary = json.loads((root / 'summary.json').read_text())
records = [json.loads(x) for x in (root / 'steps.jsonl').read_text().splitlines()]
with tarfile.open(root / 'source_snapshot.tar.gz') as archive:
    frozen = yaml.safe_load(archive.extractfile('configs/experiments.yaml').read())
key = register_lbf(3, frozen['environment'])
env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None, seed=0,
                      common_reward=True, reward_scalarisation='sum')
results = []
try:
    for episode in summary.get('episodes', []):
        seed = episode['seed']
        env.reset(seed=seed)
        steps = [item for item in records if item['seed'] == seed]
        assert len(steps) == episode['length']
        reward_sum = 0.
        for t, item in enumerate(steps):
            assert item['t'] == t and state(env) == item['before']
            image = np.asarray(Image.open(root / f'episode_{seed}_step_{t:03d}.png').convert('RGB'))
            assert np.array_equal(image, env.render()), 'Saved teacher image mismatches real replay state'
            probs = np.asarray(item['probabilities'])
            assert probs.shape == (3, 6) and np.allclose(probs.sum(-1), 1)
            assert probs.argmax(-1).tolist() == item['actions']
            _, reward, terminated, truncated, info = env.step(item['actions'])
            reward = float(np.asarray(reward).sum())
            reward_sum += reward
            assert state(env) == item['after'] and info == item['info']
            assert np.isclose(reward, item['reward'])
            assert bool(terminated or truncated) == (t == len(steps)-1)
        assert bool(info['success']) == episode['success']
        assert int(info['remaining_food']) == episode['remaining_food']
        assert np.isclose(reward_sum, episode['team_return'])
        results.append({'seed': seed, 'steps': len(steps), 'success': episode['success'],
                        'states_images_actions_rewards_and_termination_match': True})
finally:
    env.close()
result = {'status': 'passed', 'formal': False, 'completed_episodes_replayed': results,
          'note': 'Only completed episodes were replayed; in-progress episodes are not scored.'}
(root / 'replay_validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
