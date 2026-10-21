"""Adapter boundaries and retained command/import contracts; no real API."""
import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from vlm4coop.envs.base import TeacherSpec
from vlm4coop.envs.registry import get_adapter, register_adapter
from vlm4coop.training.configuration import load_config, make_args
from vlm4coop.training.setup import setup_runner
from vlm4coop.training.annotation import annotate
from vlm4coop.teachers.api import OpenCodeTeacher, build_api_cache_identity
from vlm4coop.teachers.interfaces import TeacherResult


class ToyEnv:
    """Test-only EPyMARL environment with five actions and no LBF fields."""
    episode_limit=4
    def __init__(self,seed,**kwargs): self.t=0; self.seed=seed
    def reset(self,seed=None): self.t=0
    def get_obs(self): return [np.full(4,self.t,dtype=np.float32)]*2
    def get_state(self): return np.full(8,self.t,dtype=np.float32)
    def get_avail_actions(self): return [[1]*5]*2
    def get_env_info(self): return dict(n_agents=2,n_actions=5,obs_shape=4,state_shape=8,episode_limit=4)
    def step(self,actions):
        self.t+=1
        return self.get_obs(),.25,self.t==4,False,dict(goal_achieved=self.t==4,episode_limit=False)
    def get_stats(self): return {}
    def close(self): pass
    def render(self): return np.zeros((3,3,3),dtype=np.uint8)


class ToyAdapter:
    name='refactor_test_only'
    def epymarl_args(self,n_agents,seed,config):
        return dict(env='refactor_test_only',n_actions=5,env_args=dict(seed=seed))
    def prepare(self,n_agents,config):
        from envs import REGISTRY
        REGISTRY[self.name]=ToyEnv
    def validate_info(self,args,info): assert info['n_actions']==5 and info['obs_shape']==4
    def episode_fields(self,info):
        return dict(success=info['goal_achieved'],episode_limit=info['episode_limit'],test_task=True)
    def teacher_spec(self):
        return TeacherSpec(('WAIT','UP','DOWN','LEFT','RIGHT'),lambda n:f'TEST system {n}',
            lambda n:f'TEST user {n}',lambda:dict(test_environment=self.name,renderer='test-only',actions=5))


@pytest.fixture
def toy(monkeypatch):
    from vlm4coop.envs import registry
    monkeypatch.setattr(registry,'_ADAPTERS',{})
    adapter=ToyAdapter(); register_adapter(adapter.name,adapter)
    config=deepcopy(load_config()); config['environment_adapter']=adapter.name; config['batch_size_run']=2
    return adapter,config


def test_adapter_selection_explicit_and_duplicate_guard(toy):
    adapter,config=toy
    assert get_adapter(config) is adapter
    assert get_adapter().name=='lbf'
    with pytest.raises(ValueError,match='Unregistered'): get_adapter({'environment_adapter':'mpe_not_implemented'})
    with pytest.raises(ValueError,match='already registered'): register_adapter(adapter.name,ToyAdapter())


def test_runner_batch_and_mask_do_not_require_lbf(toy):
    adapter,config=toy
    args=make_args(config,2,0,'smoke')
    logger=SimpleNamespace(log_stat=lambda *args:None)
    runner,_,_=setup_runner(args,logger,config)
    try:
        runner.capture_frames=True; runner.next_episode_seeds=[10,11]
        with torch.no_grad(): batch=runner.run(test_mode=False)
        assert runner.t_env==8 and batch['teacher_probs'].shape==(2,5,2,5)
        assert all(row['test_task'] and row['success'] and 'remaining_food' not in row for row in runner.last_episode_records)
        reward=batch['reward'].clone()
        class FailedTeacher:
            def predict_many(self,images,n_agents):
                return [TeacherResult(None,'TEST ONLY',False,source='format_failure',format_attempts=3)]*len(images)
        annotate(runner,batch,FailedTeacher())
        assert torch.equal(reward,batch['reward']) and not batch['teacher_valid'].any()
        assert batch['teacher_probs'].shape[-1]==5 and not batch['teacher_probs'].any()
    finally: runner.close_env()


def test_teacher_protocol_uses_adapter_prompts_and_action_count(toy,monkeypatch,tmp_path):
    adapter,config=toy; spec=adapter.teacher_spec()
    monkeypatch.setenv('OPENCODE_API_KEY','TEST-ONLY')
    teacher=OpenCodeTeacher(config['teacher'],tmp_path/'calls.jsonl',protocol=spec)
    class Response:
        status,headers=200,{}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            return json.dumps({'choices':[{'message':{'content':json.dumps({'agents':[
                {'id':i,'probs':[.1,.2,.2,.2,.3]} for i in range(2)]})}}]}).encode()
    def open_fake(request,**kwargs):
        payload=json.loads(request.data)
        assert payload['messages'][0]['content']=='TEST system 2'
        assert payload['messages'][1]['content'][0]['text']=='TEST user 2'
        return Response()
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen',open_fake)
    result=teacher.predict(Image.new('RGB',(3,3)),2,use_cache=False)
    assert result.probs.shape==(2,5) and teacher.calls==1
    identity=build_api_cache_identity(config['teacher'],spec)
    assert identity['test_environment']==adapter.name and 'prompt_templates' not in identity


def test_pure_none_fallback_uses_protocol_action_dimension(toy):
    from vlm4coop.evaluation.pure_vlm import TeacherMAC
    adapter,_=toy
    class FailedTeacher:
        protocol=adapter.teacher_spec()
        def predict_many(self,images,n_agents,use_cache=True):
            return [TeacherResult(None,'TEST ONLY',False,source='format_failure',format_attempts=3)]*len(images)
    mac=TeacherMAC(FailedTeacher(),2)
    mac.runner=SimpleNamespace(frames={(0,0):np.zeros((3,3,3),dtype=np.uint8)})
    assert mac.n_actions==5 and mac.select_actions(None,0,0,[0],True).tolist()==[[0,0]]


@pytest.mark.parametrize('old,new',[
    ('api_teacher','teachers.api'),('config','training.configuration'),('runner','training.runner'),
    ('environment','envs.lbf.adapter'),('rendering','envs.lbf.rendering'),('learner','algorithms.mappo'),
    ('distillation','algorithms.distillation'),('pure_vlm','evaluation.pure_vlm'),('metrics','evaluation.metrics')])
def test_only_canonical_module_is_active_old_facade_is_backed_up(old,new):
    assert importlib.util.find_spec('vlm4coop.'+old) is None
    assert importlib.import_module('vlm4coop.'+new) is not None
    source=Path(__file__).resolve().parents[1]/'src'
    assert (source/'old/vlm4coop'/f'{old}.py').is_file()


def test_dependency_direction_and_no_flat_internal_imports():
    root=Path(__file__).resolve().parents[1]/'src/vlm4coop'
    forbidden={'teachers':('training','evaluation','algorithms'),'envs':('training','evaluation','algorithms'),
               'algorithms':('training','evaluation','envs','teachers'),'utils':('training','evaluation','envs','teachers','algorithms')}
    flat={'api_teacher','teacher','environment','rendering','config','runner','learner','distillation','metrics','analysis','pure_vlm','provenance','scenario_sampling'}
    for package in ('envs','teachers','algorithms','training','evaluation','utils','cli'):
        for path in (root/package).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node,ast.ImportFrom) or not node.module or not node.module.startswith('vlm4coop.'): continue
                target=node.module.split('.')[1]
                assert target not in forbidden.get(package,()),(path,node.module)
                assert target not in flat,(path,node.module)
    # The evaluator owns no dependency on the training loop.
    assert 'training.loop' not in (root/'evaluation/evaluator.py').read_text()


@pytest.mark.parametrize('command',[
    ['-m','vlm4coop.training.loop','--help'],['-m','vlm4coop.evaluation.pure_vlm','--help'],
    ['-m','vlm4coop.training.scheduling','--help'],
    ['VLM4Coop/scripts/probes/api_bulk_benchmark.py','--help'],
    ['VLM4Coop/scripts/analysis/export_training_run.py','--help']])
def test_canonical_cli_help_without_launch(command):
    completed=subprocess.run([sys.executable,*command],capture_output=True,text=True,timeout=25)
    assert completed.returncode==0,completed.stderr
    assert 'usage:' in completed.stdout.lower()


def test_source_snapshots_exclude_old_backups(monkeypatch,tmp_path):
    import tarfile
    from vlm4coop.utils import provenance
    project=tmp_path/'VLM4Coop'
    files={'src/vlm4coop/__init__.py':'', 'src/old/retired.py':'# backup only',
           'configs/test.yaml':'test: true', 'reports/USER_PROTOCOL_DECISION.md':'test only'}
    for name,text in files.items():
        path=project/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
    (tmp_path/'AGENTS.md').write_text('test only')
    (tmp_path/'VLM4Coop_Experiment_Plan.md').write_text('test only')
    monkeypatch.setattr(provenance,'PROJECT',project)
    monkeypatch.setattr(provenance,'ROOT',tmp_path)
    output=tmp_path/'snapshot'; output.mkdir()
    provenance.capture(output)
    records=json.loads((output/'provenance.json').read_text())['files']
    assert 'src/vlm4coop/__init__.py' in records
    assert not any(name.startswith('src/old/') for name in records)
    with tarfile.open(output/'source_snapshot.tar.gz') as archive:
        assert not any(name.startswith('src/old/') for name in archive.getnames())
