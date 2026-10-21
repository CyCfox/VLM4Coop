"""Current MPE protocol regression; all remote responses here are synthetic."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import pytest
import torch
from vlm4coop.teachers.reasoning_control import ReasoningGuard, ReasoningControlError
from vlm4coop.teachers.api import OpenCodeTeacher, build_api_cache_identity
from vlm4coop.utils.rgb_frames import encode_rgb, frame_image
from vlm4coop.training.configuration import load_config, make_args
from vlm4coop.utils.paths import PROJECT


def policy():
    return dict(startup_clean_responses=3,window=100,max_reasoning_responses=1,max_in_flight=20)


def test_guard_stops_majority_at_startup_without_retry(monkeypatch,tmp_path):
    monkeypatch.setenv('OPENCODE_API_KEY','SYNTHETIC-KEY')
    cfg = dict(endpoint='https://example.invalid',model='mock',thinking={'type':'disabled'},
               reasoning_effort='none',reasoning_guard=policy(),concurrency=1000)
    teacher = OpenCodeTeacher(cfg,tmp_path/'requests.jsonl')
    class Response:
        status,headers=200,{}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            return json.dumps(dict(choices=[dict(message=dict(content='OK',reasoning_content='unexpected'))])).encode()
    def respond(request,timeout):
        payload=json.loads(request.data)
        assert payload['reasoning_effort']=='none' and payload['thinking']==dict(type='disabled')
        return Response()
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen',respond)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(teacher.request,'x') for _ in range(5)]
        for future in futures:
            with pytest.raises(ReasoningControlError): future.result(timeout=10)
    assert teacher.calls==1 and teacher.responses_with_reasoning==1
    audit=json.loads((tmp_path/'requests.jsonl').read_text())
    assert 'reasoning_guard_error' in audit


def test_guard_allows_rare_valid_reasoning_and_stops_second_in_window():
    guard=ReasoningGuard(policy())
    for _ in range(3): guard.observe('reported_zero')
    guard.observe('returned_reasoning')
    for _ in range(98): guard.observe('reported_zero')
    with pytest.raises(ReasoningControlError): guard.observe('returned_reasoning')
    assert guard.error


def test_guard_rolling_window_expires_old_reasoning():
    guard=ReasoningGuard(policy())
    for _ in range(3): guard.observe('reported_zero')
    guard.observe('returned_reasoning')
    for _ in range(99): guard.observe('reported_zero')
    guard.observe('returned_reasoning')
    assert guard.error is None


def test_lossless_frame_pixels_cache_and_operational_identity(tmp_path):
    from vlm4coop.teachers.cache import ProbabilityCache
    rgb=np.random.default_rng(2).integers(0,256,(71,105,3),dtype=np.uint8)
    encoded=frame_image(encode_rgb(rgb))
    np.testing.assert_array_equal(encoded.convert('RGB'),rgb)
    buffer=io.BytesIO(); encoded.save(buffer,format='PNG')
    np.testing.assert_array_equal(Image.open(buffer),rgb)
    config=load_config(PROJECT/'configs/mpe_spread.yaml')
    identity=build_api_cache_identity(config['teacher'])
    changed=deepcopy(config['teacher']); changed['reasoning_guard']['window']=200
    assert build_api_cache_identity(changed)==identity
    changed['top_p']=.99
    assert build_api_cache_identity(changed)!=identity
    cache=ProbabilityCache(tmp_path/'cache.sqlite',identity)
    try: assert cache.key(encoded,3)==cache.key(Image.fromarray(rgb),3)
    finally: cache.close()


def test_fixed_square_full_scene_multiple_seeds_and_moving_trajectories():
    from vlm4coop.envs.mpe.adapter import MPEGymBridge
    from vlm4coop.envs.mpe.rendering import render_geometry
    env=MPEGymBridge()
    try:
        for seed in (7,1000000,1000001,1000002):
            env.reset(seed=seed)
            for t in range(25):
                geometry=render_geometry(env.world)
                assert env.render().shape==(1440,1440,3)
                for entity in geometry['agents']+geometry['goals']:
                    p=np.asarray(entity.get('start',entity.get('center'))); r=entity['radius']
                    assert (p-r>=1-1e-7).all() and (p+r<=1439+1e-7).all()
                env.step([1,2,3])
    finally: env.close()


def test_real_mpe40_full_rollout_and_soft_hard_updates():
    from vlm4coop.training.setup import setup_runner
    from vlm4coop.training.annotation import annotate
    from vlm4coop.envs.scenario_sampling import TrainingSeedPool
    from vlm4coop.algorithms.mappo import CoopPPOLearner
    from learners.ppo_learner import PPOLearner
    from vlm4coop.teachers.interfaces import TeacherResult
    config=load_config(PROJECT/'configs/mpe_spread.yaml')
    args=make_args(config,3,0,'smoke'); torch.manual_seed(0); torch.set_num_threads(2)
    logger=SimpleNamespace(log_stat=lambda *args: None)
    runner,mac,scheme=setup_runner(args,logger,config)
    try:
        sampler=TrainingSeedPool(config); runner.next_episode_seeds=sampler.sample()
        assert len(set(runner.next_episode_seeds))==40
        runner.capture_frames=True
        with torch.no_grad(): batch=runner.run()
        assert runner.t_env==1000 and batch['obs'].shape==(40,26,3,14)
        assert len(runner.frames)==1000 and all(isinstance(f,bytes) for f in runner.frames.values())
        assert sum(map(len,runner.frames.values()))<100_000_000
        assert all(r['length']==25 and r['episode_limit'] for r in runner.last_episode_records)
        original_reward=batch['reward'].clone()
        class Teacher:
            calls=0
            def predict(self,image,n_agents):
                self.calls+=1
                return TeacherResult(np.tile([.05,.05,.8,.05,.05],(3,1)).astype(np.float32),'SYNTHETIC',False)
        teacher=Teacher(); annotate(runner,batch,teacher)
        assert teacher.calls==1000 and not runner.frames and int(batch['teacher_valid'].sum())==1000
        assert torch.equal(original_reward,batch['reward'])
        batch=batch[:,:int(batch.max_t_filled())]
        for mode in ('soft','hard'):
            variant=deepcopy(args); variant.kd_mode=mode; variant.kd_lambda_current=.1
            learner=CoopPPOLearner(deepcopy(mac),scheme,logger,variant)
            learner.train(batch,1000,0)
            assert learner.args.epochs==4
            assert any(not torch.equal(a,b) for a,b in zip(mac.parameters(),learner.mac.parameters()))
    finally: runner.close_env()


def test_real_mappo40_one_update_then_one_evaluation(monkeypatch,tmp_path):
    from vlm4coop.training import loop
    from learners.ppo_learner import PPOLearner
    from vlm4coop.envs.scenario_sampling import audit_seed_pool
    config=load_config(PROJECT/'configs/mpe_spread.yaml')
    config['protocol_document']=str(PROJECT/config['protocol_document'])
    monkeypatch.setattr(loop,'PROJECT',tmp_path)
    original=PPOLearner.train; calls=[]
    def observed(self,batch,t_env,episode_num):
        calls.append((t_env,episode_num,self.args.epochs,batch.batch_size))
        return original(self,batch,t_env,episode_num)
    monkeypatch.setattr(PPOLearner,'train',observed)
    result=loop.train('mappo',3,0,'smoke',tmp_path/'engineering',config=config)
    assert result['status']=='complete' and result['train_env_steps']==1000 and result['episodes']==40
    assert result['teacher_calls']==0 and calls==[(1000,0,4,40)]
    run=tmp_path/'engineering'
    evaluations=[json.loads(r) for r in (run/'evaluations.jsonl').read_text().splitlines()]
    assert len(evaluations)==1 and evaluations[0]['actual_train_env_steps']==1000
    assert len((run/'eval_episodes.jsonl').read_text().splitlines())==100
    assert audit_seed_pool(run)['status']=='passed'


def test_real_pure_mpe_lossless_frame_transport_and_saved_images(monkeypatch,tmp_path):
    from vlm4coop.evaluation import pure_vlm
    from vlm4coop.envs.registry import get_adapter
    from vlm4coop.teachers.interfaces import TeacherResult
    config=load_config(PROJECT/'configs/mpe_spread.yaml')
    config.update(eval_parallel_envs=3,pure_vlm_parallel_envs=3,eval_episodes_every_checkpoint=3)
    config['fixed_eval_seeds']['count']=3
    spec=get_adapter(config).teacher_spec()
    class Synthetic:
        config={'pure_vlm_use_cache':False}; protocol=spec; cache=None; calls=0
        def predict_many(self,images,n_agents,use_cache=True):
            assert not use_cache
            assert all(image.convert('RGB').size==(1440,1440) for image in images)
            self.calls+=len(images)
            return [TeacherResult(np.tile([1,0,0,0,0],(3,1)).astype(np.float32),'SYNTHETIC',False)]*len(images)
    teacher=Synthetic()
    monkeypatch.setattr(pure_vlm,'make_teacher',lambda *args,**kwargs:teacher)
    monkeypatch.setattr(pure_vlm,'PROJECT',tmp_path)
    run=tmp_path/'engineering_pure'
    result=pure_vlm.run(run,config=config,profile='smoke')
    assert result['n_eval']==3 and result['decisions']==75 and teacher.calls==75
    files=list((run/'frames').glob('*.png'))
    assert len(files)==75 and result['cache_hits']==0
    for file in files:
        with Image.open(file) as image: assert image.size==(1440,1440)
