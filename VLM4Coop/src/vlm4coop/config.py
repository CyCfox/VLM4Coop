from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import yaml
from vlm4coop.environment import lbf_key

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "VLM4Coop"


def load_config(path=None):
    with open(path or PROJECT / "configs/experiments.yaml") as stream:
        return yaml.safe_load(stream)


def make_args(config, n_agents, seed, profile="formal"):
    data = {}
    for name in ("default.yaml", "algs/mappo.yaml", "envs/gymma.yaml"):
        with (ROOT / "epymarl/src/config" / name).open() as stream:
            data.update(yaml.safe_load(stream))
    for key in ("runner", "batch_size_run", "batch_size", "buffer_size", "epochs", "test_greedy"):
        data[key] = config[key]
    budget = {"formal": "formal_train_steps", "smoke": "smoke_train_steps", "pilot": "pilot_train_steps"}[profile]
    data.update(n_agents=n_agents, n_actions=6, seed=seed, t_max=config[budget],
                env="vlm4coop", device="cpu", use_cuda=False, common_reward=True,
                reward_scalarisation="sum", test_nepisode=config["eval_episodes_every_checkpoint"],
                test_interval=config["eval_interval_steps"], render=False,
                env_args={"key": lbf_key(n_agents, config['environment']),
                          "seed": seed, "time_limit": config["environment"]["max_episode_steps"],
                          "pretrained_wrapper": None})
    return SimpleNamespace(**deepcopy(data))
