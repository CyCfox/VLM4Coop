"""Formal Soft configuration and real-native cutoff regression without API calls."""
from copy import deepcopy
import json
import numpy as np
import torch
from vlm4coop.training.configuration import load_config, make_args
from vlm4coop.utils.paths import PROJECT


def test_formal_soft_args_cache_identity_and_cutoff():
    from vlm4coop.cli.diagnostics.record_mpe_soft import verify_config
    from vlm4coop.algorithms.distillation import kd_weight
    c = load_config(PROJECT/'configs/mpe_soft1000.yaml')
    frozen = dict(experiment_config=c, epymarl_args=vars(make_args(c,3,0,'formal')),
        method='soft', agents=3, seed=0, profile='formal')
    verify_config(frozen)
    assert load_config(PROJECT/'configs/mpe_spread.yaml')['formal_authorized_methods'] == []
    assert kd_weight(99000,c['kd_lambda_initial'],c['kd_lambda_plateau'],c['kd_anneal_t1'],c['kd_anneal_t2']) == .1
    assert kd_weight(100000,c['kd_lambda_initial'],c['kd_lambda_plateau'],c['kd_anneal_t1'],c['kd_anneal_t2']) == 0
    assert kd_weight(1000000,c['kd_lambda_initial'],c['kd_lambda_plateau'],c['kd_anneal_t1'],c['kd_anneal_t2']) == 0


def test_real_soft_cutoff_keeps_PPO_and_evaluation(monkeypatch,tmp_path):
    """Scale only engineering cutoff to2k; actual40x25physics/4epochs remain real."""
    from vlm4coop.training import loop
    from vlm4coop.algorithms.mappo import CoopPPOLearner
    from vlm4coop.teachers.interfaces import TeacherResult
    from vlm4coop.envs.scenario_sampling import audit_seed_pool
    c = load_config(PROJECT/'configs/mpe_soft1000.yaml')
    c.update(smoke_train_steps=3000,kd_anneal_t1=1000,kd_anneal_t2=2000)
    c['protocol_document'] = str(PROJECT/c['protocol_document'])
    monkeypatch.setattr(loop,'PROJECT',tmp_path)
    class Synthetic:
        config={'concurrency':1000}; calls=0; cache=None
        def predict_many(self, images, n_agents):
            assert len(images)==1000 and n_agents==3
            assert all(im.convert('RGB').size==(1440,1440) for im in images[:2])
            self.calls+=len(images)
            return [TeacherResult(np.tile([.05,.05,.8,.05,.05],(3,1)).astype(np.float32),'SYNTHETIC',False)]*len(images)
    teacher=Synthetic(); events=[]; updates=[]; captures=[]
    original=CoopPPOLearner.train
    def observed(self,batch,t_env,episode_num):
        before=[p.detach().clone() for p in self.mac.parameters()]
        events.append(('update',t_env))
        updates.append((t_env,self.args.epochs,self.args.kd_lambda_current,int(batch['teacher_valid'].sum())))
        result=original(self,batch,t_env,episode_num)
        assert any(not torch.equal(a,b) for a,b in zip(before,self.mac.parameters()))
        return result
    monkeypatch.setattr(CoopPPOLearner,'train',observed)
    evaluate=loop.evaluate
    def checked_evaluation(*args,**kwargs):
        calls=teacher.calls
        result=evaluate(*args,**kwargs)
        assert teacher.calls==calls
        events.append(('evaluate',result['actual_train_env_steps']))
        return result
    monkeypatch.setattr(loop,'evaluate',checked_evaluation)
    from vlm4coop.training.runner import CoopParallelRunner
    native_run=CoopParallelRunner.run
    def observed_run(self,*args,**kwargs):
        if not kwargs.get('test_mode',False): captures.append((self.t_env,self.capture_frames))
        return native_run(self,*args,**kwargs)
    monkeypatch.setattr(CoopParallelRunner,'run',observed_run)
    run=tmp_path/'engineering_soft_cutoff'
    result=loop.train('soft',3,0,'smoke',run,config=c,teacher=teacher)
    assert result['status']=='complete' and result['train_env_steps']==3000 and result['episodes']==120
    assert updates==[(1000,4,.1,1000),(2000,4,0,0),(3000,4,0,0)]
    assert captures==[(0,True),(1000,True),(2000,False)]
    assert events==[(kind,t) for t in (1000,2000,3000) for kind in ('update','evaluate')]
    assert teacher.calls==1000 and len((run/'eval_episodes.jsonl').read_text().splitlines())==300
    assert audit_seed_pool(run)['status']=='passed'
