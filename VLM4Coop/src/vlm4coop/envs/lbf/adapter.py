"""Public LBF registration and a minimal EPyMARL Gymma subclass."""
import gymnasium as gym
import numpy as np
import lbforaging  # noqa: F401, registers original tasks without modifying them
from envs.gymma import GymmaWrapper
from vlm4coop.envs.lbf.rendering import global_rgb


def lbf_key(n_agents, config):
    rows, cols = config['field_size']
    return f"VLM4Coop-{rows}x{cols}-{n_agents}p-{n_agents}f-v0"


def register_lbf(n_agents, config):
    key = lbf_key(n_agents, config)
    kwargs = dict(config, players=n_agents, max_num_food=n_agents)
    kwargs["field_size"] = tuple(kwargs["field_size"])
    if key not in gym.registry:
        gym.register(key, entry_point="lbforaging.foraging:ForagingEnv", kwargs=kwargs,
                     disable_env_checker=True)
    elif gym.registry[key].kwargs != kwargs:
        raise ValueError("Conflicting registration for frozen LBF task")
    return key


class CoopGymmaWrapper(GymmaWrapper):
    def reset(self, seed=None, options=None):
        if seed is not None:
            # LBF 2.0 spawn_players otherwise rejects positions retained from
            # the previous episode. Clear public position fields to make a
            # fixed seeded scenario independent of the prior policy's end state.
            for player in self._env.unwrapped.players:
                player.position = None
        obs, info = super().reset(seed=seed, options=options)
        self.initial_food_count = int(np.count_nonzero(self._env.unwrapped.field))
        if self.initial_food_count != self.n_agents:
            raise ValueError("LBF failed to spawn requested food count; preserve logs and inspect seed")
        # Row-major IDs are assigned once per episode, so collecting F1 does
        # not rename the remaining foods. This bookkeeping is rendering only.
        self.food_ids = {tuple(position): food_id for food_id, position in
                         enumerate(np.argwhere(self._env.unwrapped.field > 0))}
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        raw = self._env.unwrapped
        remaining = int(np.count_nonzero(raw.field))
        success = remaining == 0 and self.initial_food_count > 0
        # LBF 2.0 reports internal time limits as terminated; tell the native
        # EPyMARL runner whether this is a time limit, preserving its mask logic.
        info = dict(info, success=int(success), remaining_food=remaining,
                    initial_food_count=self.initial_food_count,
                    episode_limit=bool((terminated or truncated) and not success
                                       and raw.current_step >= self.episode_limit))
        return obs, reward, terminated, truncated, info

    def render(self):
        return global_rgb(self._env.unwrapped, self.food_ids)

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]


def make_env(**kwargs):
    return CoopGymmaWrapper(**kwargs)


class LBFAdapter:
    """Own LBF-specific registration, dimensions, episode fields and teacher I/O."""
    name = 'lbf'

    def epymarl_args(self, n_agents, seed, config):
        return dict(env='vlm4coop', n_actions=6,
                    env_args=dict(key=lbf_key(n_agents, config), seed=seed,
                                  time_limit=config['max_episode_steps'], pretrained_wrapper=None))

    def prepare(self, n_agents, config):
        from envs import REGISTRY
        register_lbf(n_agents, config)
        REGISTRY['vlm4coop'] = make_env

    def validate_info(self, args, info):
        assert info['obs_shape'] == 6 * args.n_agents, info

    def episode_fields(self, info):
        return dict(success=bool(info['success']), initial_food_count=info['initial_food_count'],
                    remaining_food=info['remaining_food'],
                    episode_limit=bool(info.get('episode_limit', False)))

    def teacher_spec(self):
        from vlm4coop.envs.base import TeacherSpec
        from vlm4coop.envs.lbf.prompts import ACTIONS, api_system_prompt, api_joint_prompt
        return TeacherSpec(ACTIONS, api_system_prompt, api_joint_prompt, self.teacher_cache_identity)

    @staticmethod
    def teacher_cache_identity():
        from vlm4coop.envs.lbf.prompts import API_SYSTEM_PROMPT, api_system_prompt, api_joint_prompt
        from vlm4coop.envs.lbf.rendering import RENDERER_VERSION
        # Exact historical fields and values: old persisted LBF keys remain valid.
        return dict(renderer=RENDERER_VERSION, system_prompt=API_SYSTEM_PROMPT,
                    system_prompt_templates={n: api_system_prompt(n) for n in (3, 4, 5)},
                    prompt_templates={n: api_joint_prompt(n) for n in (3, 4, 5)})
