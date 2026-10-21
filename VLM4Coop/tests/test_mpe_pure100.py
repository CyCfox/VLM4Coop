"""Formal Pure100 configuration and actual client concurrency; zero real API."""
from copy import deepcopy
import json
import threading
import pytest
from PIL import Image
from vlm4coop.training.configuration import load_config
from vlm4coop.utils.paths import PROJECT
from vlm4coop.envs.registry import get_adapter
from vlm4coop.teachers.api import OpenCodeTeacher,build_api_cache_identity


def test_pure100_only_authorized_and_training_unchanged():
    base=load_config(PROJECT/'configs/mpe_spread.yaml')
    cfg=load_config(PROJECT/'configs/mpe_pure_vlm100.yaml')
    assert base['formal_authorized_methods']==[] and cfg['formal_authorized_methods']==['pure_vlm']
    assert cfg['pure_vlm_parallel_envs']==cfg['pure_vlm_concurrency']==100
    assert cfg['teacher']['reasoning_guard']['max_in_flight']==100
    assert base['teacher']['reasoning_guard']['max_in_flight']==20
    for key in ('environment','training_seed_pool','fixed_eval_seeds','batch_size_run','batch_size',
                'buffer_size','epochs','formal_train_steps','eval_interval_steps','test_greedy'):
        assert cfg[key]==base[key]
    assert cfg['teacher']['pure_vlm_use_cache'] is False
    spec=get_adapter(cfg).teacher_spec()
    assert build_api_cache_identity(cfg['teacher'],spec)==build_api_cache_identity(base['teacher'],spec)


def test_guarded_pure_batch_reaches_100_HTTP_slots_without_cache(monkeypatch,tmp_path):
    cfg=load_config(PROJECT/'configs/mpe_pure_vlm100.yaml')
    tc=deepcopy(cfg['teacher']);tc['concurrency']=cfg['pure_vlm_concurrency']
    monkeypatch.setenv('OPENCODE_API_KEY','SYNTHETIC-TEST-ONLY')
    teacher=OpenCodeTeacher(tc,tmp_path/'requests.jsonl',purpose='pure',protocol=get_adapter(cfg).teacher_spec())
    lock=threading.Lock(); count=0
    startup_barrier=threading.Barrier(97); full_barrier=threading.Barrier(100)
    class Response:
        status,headers=200,{}
        def __init__(self,index): self.index=index
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            if 3 < self.index <=100: startup_barrier.wait(timeout=30)
            elif self.index>100: full_barrier.wait(timeout=30)
            return json.dumps(dict(id=f'TEST_{self.index}',model='SYNTHETIC',
                choices=[dict(message=dict(content=json.dumps(dict(agents=[dict(id=i,probs=[1,0,0,0,0]) for i in range(3)])),reasoning_content=''))],
                usage=dict(completion_tokens_details=dict(reasoning_tokens=0)))).encode()
    def open_fake(request,timeout):
        nonlocal count
        payload=json.loads(request.data)
        assert payload['thinking']==dict(type='disabled')
        assert payload['chat_template_kwargs']==dict(thinking=False,enable_thinking=False)
        assert 'reasoning_effort' not in payload
        with lock: count+=1;index=count
        return Response(index)
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen',open_fake)
    # Identical images still require distinct calls in uncached Pure VLM.
    images=[Image.new('RGB',(3,3))]*100
    for _ in range(2):
        result=teacher.predict_many(images,3,use_cache=False)
        assert len(result)==100 and all(not r.cache_hit for r in result)
    assert teacher.calls==count==200 and teacher.max_active_requests==100
    assert teacher.cache is None and teacher.responses_with_reasoning==0


def test_native_MPE_replay_rejects_wrong_RGB_and_reward(tmp_path):
    from vlm4coop.envs.mpe.adapter import MPEGymBridge
    from vlm4coop.evaluation.mpe_replay import replay_pure_mpe
    from PIL import Image
    cfg=load_config(PROJECT/'configs/mpe_pure_vlm100.yaml')
    env=MPEGymBridge();steps=[];reward=0.;(tmp_path/'frames').mkdir()
    try:
        env.reset(seed=1000000)
        for t in range(25):
            file=f'worker000_t{t:03d}.png'
            Image.fromarray(env.render()).save(tmp_path/'frames'/file)
            _,r,_,_,info=env.step([0,0,0]);reward+=sum(r)
            steps.append(dict(env_seed=1000000,worker=0,t=t,actions=[0,0,0]))
        episode=dict(get_adapter(cfg).episode_fields(info),env_seed=1000000,team_return=reward)
        assert replay_pure_mpe(tmp_path,cfg,[episode],steps)['RGB_frames_replayed']==25
        wrong=dict(episode,team_return=reward+1)
        with pytest.raises(AssertionError): replay_pure_mpe(tmp_path,cfg,[wrong],steps)
        Image.new('RGB',(1440,1440)).save(tmp_path/'frames/worker000_t000.png')
        with pytest.raises(AssertionError): replay_pure_mpe(tmp_path,cfg,[episode],steps)
    finally: env.close()
