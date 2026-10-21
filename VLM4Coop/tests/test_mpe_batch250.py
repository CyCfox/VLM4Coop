"""Real short MPE training and figure accounting; no teacher API or paper results."""
from copy import deepcopy
import json
import runpy
from shutil import copytree
import sys
import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip('pettingzoo')
pytest.importorskip('matplotlib')
from vlm4coop.training.configuration import load_config, make_args
from vlm4coop.utils.paths import PROJECT
from vlm4coop.evaluation.analysis import read_jsonl
from vlm4coop.evaluation.plotting import plot_training_curves


def test_mpe_formal_standard_updates1000_evaluates1000_and_is_paused():
    config = load_config(PROJECT/'configs/mpe_spread.yaml')
    assert config['batch_size_run'] == config['batch_size'] == config['buffer_size'] == 40
    assert 40*config['environment']['max_episode_steps'] == 1000
    assert config['eval_interval_steps'] == 1000
    assert config['formal_authorized_methods'] == []
    assert config['epochs'] == 4 and config['formal_train_steps'] == 1000000
    assert config['eval_episodes_every_checkpoint'] == 100 and config['eval_parallel_envs'] == 10
    assert make_args(config,3,0).test_interval == 1000
    # The LBF protocol remains the existing 1000-step evaluation standard.
    assert load_config()['eval_interval_steps'] == 1000


def test_real_batch250_updates_evaluation_rng_and_automatic_curves(monkeypatch,tmp_path):
    from vlm4coop.training import loop
    from vlm4coop.cli.analysis import export_training_run
    from learners.ppo_learner import PPOLearner
    config = load_config(PROJECT/'configs/mpe_spread.yaml')
    config['formal_train_steps'] = 500
    config.update(batch_size_run=10,batch_size=10,buffer_size=10)  # Historical cadence regression, not current protocol.
    config['formal_authorized_methods'] = ['mappo']  # Only the explicit engineering fixture.
    config['test_fixture'] = 'ENGINEERING ONLY: shortened budget; not a formal experiment result'
    config['protocol_document'] = str(PROJECT/config['protocol_document'])
    monkeypatch.setattr(loop,'PROJECT',tmp_path)
    monkeypatch.setattr(export_training_run,'PROJECT',tmp_path)
    original_train = PPOLearner.train
    calls = []
    def observed_train(self,batch,t_env,episode_num):
        calls.append((t_env,episode_num,self.args.epochs))
        return original_train(self,batch,t_env,episode_num)
    monkeypatch.setattr(PPOLearner,'train',observed_train)
    runs = []
    for interval in (250,500):
        variant = deepcopy(config); variant['eval_interval_steps'] = interval
        run = tmp_path/f'engineering_only_every{interval}'
        status = loop.train('mappo',3,0,'formal',run,config=variant)
        assert status['train_env_steps'] == 500 and status['episodes'] == 20 and status['teacher_calls'] == 0
        expected = list(range(interval,501,interval))
        rows = read_jsonl(run/'evaluations.jsonl')
        assert [r['actual_train_env_steps'] for r in rows] == expected
        assert [r['scheduled_eval_step'] for r in rows] == expected
        episodes = read_jsonl(run/'eval_episodes.jsonl')
        assert len(episodes) == 100*len(expected)
        assert all(r['length'] == 25 for r in episodes)
        runs.append(run)
    assert calls == [(250,0,4),(500,10,4)]*2
    # Changing only eval cadence leaves native actor and critic updates identical.
    for name in ('agent.th','critic.th'):
        first = torch.load(runs[0]/'checkpoints/500'/name,map_location='cpu')
        second = torch.load(runs[1]/'checkpoints/500'/name,map_location='cpu')
        assert first.keys() == second.keys()
        for key in first: assert torch.equal(first[key],second[key]), key
    assert read_jsonl(runs[0]/'train_batches.jsonl') == read_jsonl(runs[1]/'train_batches.jsonl')
    output = tmp_path/'automatic_export'
    monkeypatch.setattr(sys,'argv',['export_training_run',str(runs[0]),'--output',str(output)])
    export_training_run.main()
    validation = json.loads((output/'validation.json').read_text())
    assert validation['status'] == 'passed' and validation['figures']['points'] == 2
    metadata = json.loads((output/'plot_metadata.json').read_text())
    assert metadata['actual_steps'] == [250,500] and metadata['smoothing'] is False
    assert metadata['evaluation_interval_steps'] == 250 and len(metadata['metrics']) == 4
    with Image.open(output/'learning_curves.png') as image:
        assert image.size[0] >= 2000 and image.size[1] >= 1000
    assert (output/'learning_curves.pdf').read_bytes().startswith(b'%PDF')
    assert '<svg' in (output/'learning_curves.svg').read_text()
    # A missing entire evaluation must fail even if the remaining row counts match.
    missing = tmp_path/'missing_eval_fixture'; copytree(runs[0],missing)
    for filename in ('evaluations.jsonl','eval_episodes.jsonl'):
        rows = [r for r in read_jsonl(missing/filename) if r['scheduled_eval_step'] != 250]
        (missing/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(sys,'argv',['validate_outputs',str(missing)])
    with pytest.raises(AssertionError):
        runpy.run_module('vlm4coop.cli.validation.validate_outputs',run_name='__main__')


def test_mpe_paused_formal_methods_are_rejected_before_output_creation(tmp_path):
    from vlm4coop.training.loop import train
    config = load_config(PROJECT/'configs/mpe_spread.yaml')
    for method in ('mappo','soft','hard'):
        output = tmp_path/method
        with pytest.raises(ValueError,match='Unresolved formal protocol'):
            train(method,3,0,'formal',output,config=config)
        assert not output.exists()


def test_plot_preserves_actual_steps_and_does_not_overwrite(tmp_path):
    config = load_config(PROJECT/'configs/mpe_spread.yaml')
    config['eval_interval_steps'] = 250  # Explicit synthetic plotting fixture cadence.
    frozen = dict(experiment_config=config,method='mappo',seed=0)
    rows = [dict(actual_train_env_steps=x,return_mean=-float(x),return_std=1.,
                 final_coverage_fraction_mean=.2,final_coverage_fraction_std=.1,
                 success_rate_mean=0.,success_rate_std=0.,
                 final_mean_min_distance_mean=.5,final_mean_min_distance_std=.1) for x in (250,500)]
    output = tmp_path/'curves'; output.mkdir()
    plot_training_curves(rows,frozen,output)
    original = (output/'learning_curves.png').read_bytes()
    with pytest.raises(FileExistsError): plot_training_curves(rows,frozen,output)
    assert (output/'learning_curves.png').read_bytes() == original
    invalid = tmp_path/'invalid'; invalid.mkdir()
    with pytest.raises(ValueError,match='increasing'): plot_training_curves([rows[0],rows[0]],frozen,invalid)
