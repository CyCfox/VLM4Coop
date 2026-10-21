"""Build native EPyMARL batches/controllers from an environment adapter."""
import torch
from components.transforms import OneHot
from controllers.basic_controller import BasicMAC
from vlm4coop.envs.registry import get_adapter
from vlm4coop.training.runner import CoopParallelRunner

def setup_runner(args, logger, config, mac=None):
    adapter = get_adapter(config)
    adapter.prepare(args.n_agents, config["environment"])
    runner = CoopParallelRunner(args, logger, episode_fields=adapter.episode_fields)
    runner.lossless_frame_transport = adapter.name == 'mpe_simple_spread'
    info = runner.get_env_info()
    adapter.validate_info(args, info)
    args.n_actions = info["n_actions"]
    scheme = {"state": {"vshape": info["state_shape"]},
              "obs": {"vshape": info["obs_shape"], "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)},
              "terminated": {"vshape": (1,), "dtype": torch.uint8},
              "teacher_probs": {"vshape": (args.n_actions,), "group": "agents"},
              "teacher_valid": {"vshape": (1,)}}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    # EpisodeBatch preprocessing adds the one-hot scheme before MAC construction.
    from components.episode_buffer import EpisodeBatch
    template = EpisodeBatch(scheme, groups, 1, info["episode_limit"] + 1, preprocess=preprocess)
    mac = mac or BasicMAC(template.scheme, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    return runner, mac, template.scheme

