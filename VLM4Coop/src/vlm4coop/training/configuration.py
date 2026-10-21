from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import yaml
from vlm4coop.envs.registry import get_adapter
from vlm4coop.utils.paths import PROJECT, ROOT



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
    adapter = get_adapter(config)
    data.update(n_agents=n_agents, seed=seed, t_max=config[budget],
                device="cpu", use_cuda=False, common_reward=True,
                reward_scalarisation="sum", test_nepisode=config["eval_episodes_every_checkpoint"],
                test_interval=config["eval_interval_steps"], render=False,
                **adapter.epymarl_args(n_agents, seed, config["environment"]))
    return SimpleNamespace(**deepcopy(data))
