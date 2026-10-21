"""Real MPE trajectory plumbing with synthetic teacher outputs only; zeroAPI."""
import json
import numpy as np
import pytest
pytest.importorskip('pettingzoo')
from vlm4coop.training.configuration import load_config
from vlm4coop.utils.paths import PROJECT
from vlm4coop.envs.registry import get_adapter
from vlm4coop.teachers.interfaces import TeacherResult
from vlm4coop.cli.probes import mpe_teacher_probe as probe
from vlm4coop.envs.mpe.prompts import system_prompt,joint_prompt,PROMPT_VERSION


def test_exact_user_mpe_prompt_and_five_actions():
    cfg=load_config(PROJECT/'configs/mpe_spread.yaml')
    assert cfg['teacher']['prompt_version']==PROMPT_VERSION
    assert system_prompt(3)=='You are a concise MPE multi-agent cooperative controller. Based on the attached figure, select a synchronous joint action for all 3 agents and output soft-probability JSON.'
    text=joint_prompt(3)
    assert '0.75×current velocity+0.1×total applied force' in text
    assert 'if a numbered landmark is not displayed, it means it has been covered.' in text
    assert 'First briefly reason in the output' in text
    assert '{"agents":[{"id":0,"probs":[five probabilities]},...]}' in text
    assert '{{' not in text and text.count('\n')==5
    with pytest.raises(ValueError): joint_prompt(4)
    assert cfg['formal_authorized_methods']==[]


def test_probe_three_real_25_step_paths_and_shared_none_fallback(monkeypatch,tmp_path):
    cfg=load_config(PROJECT/'configs/mpe_spread.yaml')
    cfg['test_fixture']='SYNTHETIC TEACHER ONLY; no model ability or real API result'
    monkeypatch.setattr(probe,'PROJECT',tmp_path)
    class SyntheticTeacher:
        calls=0; cache=None; config={}; batches=0
        protocol=get_adapter(cfg).teacher_spec()
        def predict_many(self,images,n_agents,use_cache):
            assert len(images)==3 and n_agents==3 and use_cache is False
            self.batches+=1
            p=np.zeros((3,5),dtype=np.float32); p[np.arange(3),[1,2,3]]=1
            rows=[TeacherResult(p,'SYNTHETIC TEST ONLY',False) for _ in images]
            if self.batches==1:
                rows[0]=TeacherResult(None,'SYNTHETIC FORMAT FAILURE',False,source='format_failure',format_attempts=3)
            return rows
    teacher=SyntheticTeacher()
    def fake_factory(config,output,use_cache,purpose):
        assert use_cache is False and purpose=='pure' and config['teacher']['concurrency']==3
        return teacher
    monkeypatch.setattr(probe,'make_teacher',fake_factory)
    output=tmp_path/'results/probe'
    status=probe.run_probe(output,config=cfg)
    assert status['status']=='complete' and teacher.batches==25
    assert status['completed_joint_decisions']==75 and status['teacher_calls']==0
    assert status['none_fallback_decisions']==1 and len(status['episodes'])==3
    assert all(row['length']==25 and row['episode_limit'] for row in status['episodes'])
    trajectories=[json.loads(line) for line in (output/'trajectory.jsonl').read_text().splitlines()]
    assert len(trajectories)==75 and trajectories[0]['actions']==[0,0,0]
    for worker in range(3):
        rows=[r for r in trajectories if r['worker']==worker]
        assert [r['t'] for r in rows]==list(range(25))
        assert rows[-1]['truncated'] and not rows[0]['truncated']
        assert all(len(r['after']['landmarks'])==3 for r in rows)
    assert len(list((output/'frames').glob('*.png')))==75
    assert len(list((output/'initial').glob('*.png')))==len(list((output/'final').glob('*.png')))==3
    frozen=json.loads((output/'config.json').read_text())
    assert frozen['formal'] is False and frozen['profile']=='probe'
    assert frozen['environment_seeds']==[1000000,1000001,1000002]
    assert frozen['provided_current_coordinates'] is False


def test_sample_only_three_real_resets_never_constructs_teacher(monkeypatch,tmp_path):
    cfg=load_config(PROJECT/'configs/mpe_spread.yaml')
    monkeypatch.setattr(probe,'PROJECT',tmp_path)
    def forbidden(*args,**kwargs): raise AssertionError('Sample-only must not create a teacher')
    monkeypatch.setattr(probe,'make_teacher',forbidden)
    output=tmp_path/'results/sample'
    status=probe.run_probe(output,config=cfg,sample_only=True)
    assert status['status']=='sampled_no_api' and status['teacher_calls']==0
    assert len(list((output/'initial').glob('*.png')))==3
    assert not (output/'trajectory.jsonl').exists()
