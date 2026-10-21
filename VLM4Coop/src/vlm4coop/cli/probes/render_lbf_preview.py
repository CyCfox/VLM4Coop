"""Save an actual seeded LBF reset using the same renderer as formal runs."""
import argparse
import json
from pathlib import Path

from PIL import Image
from vlm4coop.training.configuration import PROJECT, load_config
from vlm4coop.envs.lbf.adapter import CoopGymmaWrapper, register_lbf
from vlm4coop.envs.lbf.rendering import RENDERER_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents', type=int, choices=(3, 4, 5), default=3)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT.resolve()):
        raise ValueError('Output must remain under VLM4Coop')
    output.mkdir(parents=True, exist_ok=False)
    config = load_config()['environment']
    key = register_lbf(args.agents, config)
    env = CoopGymmaWrapper(key=key, time_limit=config['max_episode_steps'],
        pretrained_wrapper=None, seed=args.seed, common_reward=True, reward_scalarisation='sum')
    try:
        env.reset(seed=args.seed)
        image = Image.fromarray(env.render())
        image.save(output / 'lbf_rgb.png')
        raw = env._env.unwrapped
        metadata = dict(renderer=RENDERER_VERSION, environment=config, env_seed=args.seed,
            image_size=list(image.size), food_numbering='F0..F(n-1) assigned in row-major order at reset; stable within episode',
            agents=[dict(id=f'A{i}', row=int(p.position[0]), col=int(p.position[1]), level=int(p.level))
                    for i, p in enumerate(raw.players)],
            foods=[dict(id=f'F{food_id}', row=int(pos[0]), col=int(pos[1]), level=int(raw.field[pos]))
                   for pos, food_id in env.food_ids.items()])
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(dict(status='complete', image=str(output / 'lbf_rgb.png'),
                              renderer=RENDERER_VERSION, api_requests=0)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
