"""Audit completed Pure MPE episodes against native physics and exact RGB."""
import numpy as np
from PIL import Image


def replay_pure_mpe(run, config, episodes, steps):
    from vlm4coop.envs.mpe.adapter import MPEGymBridge
    total = 0
    ever = 0
    by_seed = {}
    for step in steps:
        by_seed.setdefault(step['env_seed'], []).append(step)
    for episode in episodes:
        env = MPEGymBridge(**config['environment'])
        reward_sum = 0.
        try:
            env.reset(seed=episode['env_seed'])
            rows = by_seed[episode['env_seed']]
            assert [row['t'] for row in rows] == list(range(25))
            for row in rows:
                frame = row.get('frame_file', f"worker{row['worker']:03d}_t{row['t']:03d}.png")
                with Image.open(run/'frames'/frame) as image:
                    assert image.size == (1440,1440)
                    np.testing.assert_array_equal(np.asarray(image.convert('RGB')), env.render())
                _, rewards, terminated, truncated, info = env.step(row['actions'])
                assert not terminated and bool(truncated) == (row['t'] == 24)
                reward_sum += sum(rewards)
                total += 1
            assert abs(reward_sum-episode['team_return']) < 1e-9
            assert info['covered_landmarks'] == episode['final_covered_landmarks']
            assert bool(info['success']) == episode['success']
            assert info['collision_pairs_total'] == episode['collision_pairs_total']
            assert info['first_all_covered_step'] == episode['first_all_covered_step']
            assert abs(info['mean_min_distance']-episode['final_mean_min_distance']) < 1e-12
            assert abs(info['sum_min_distance']-episode['final_sum_min_distance']) < 1e-12
            ever += int(info['first_all_covered_step'] >= 0)
        finally:
            env.close()
    return dict(status='passed', native_steps_replayed=total, RGB_frames_replayed=total,
                image_size=[1440,1440], ever_all_covered_episodes=ever,
                episode_returns_final_coverage_and_collisions_verified=True)
