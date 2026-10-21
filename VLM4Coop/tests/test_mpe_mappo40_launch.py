"""MAPPO40 launch authorization and unchanged native student arguments."""
import json
from vlm4coop.training.configuration import load_config,make_args
from vlm4coop.utils.paths import PROJECT
from vlm4coop.envs.scenario_sampling import TrainingSeedPool


def test_mappo40_formal_config_only_authorizes_requested_method():
    base=load_config(PROJECT/'configs/mpe_spread.yaml')
    cfg=load_config(PROJECT/'configs/mpe_mappo40.yaml')
    pure=load_config(PROJECT/'configs/mpe_pure_vlm100.yaml')
    assert cfg['formal_authorized_methods']==['mappo']
    assert base['formal_authorized_methods']==[] and pure['formal_authorized_methods']==['pure_vlm']
    assert cfg['batch_size_run']==cfg['batch_size']==cfg['buffer_size']==40
    assert cfg['epochs']==4 and cfg['formal_train_steps']==1000000 and cfg['eval_interval_steps']==1000
    for profile in ('smoke','formal'):
        assert vars(make_args(cfg,3,0,profile))==vars(make_args(base,3,0,profile))
    old=json.loads((PROJECT/'results/mappo_mpe_spread_fixed500_1000k_20261020_01/config.json').read_text())['epymarl_args']
    now=vars(make_args(cfg,3,0,'formal'))
    assert sorted(k for k in now if now[k]!=old.get(k))==['batch_size','batch_size_run','buffer_size']
    sampler=TrainingSeedPool(cfg)
    assert len(sampler.seeds)==500 and len(set(sampler.sample()))==40
