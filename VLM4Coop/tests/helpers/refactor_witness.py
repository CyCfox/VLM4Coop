"""Deterministic pre/post-refactor witness; real LBF, synthetic teachers, no API."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from learners.ppo_learner import PPOLearner
from vlm4coop.training.configuration import load_config, make_args
from vlm4coop.envs.lbf.adapter import register_lbf, CoopGymmaWrapper
from vlm4coop.training.setup import setup_runner
from vlm4coop.training.annotation import annotate
from vlm4coop.evaluation.evaluator import evaluate
from vlm4coop.utils.json_logging import _json_safe
from vlm4coop.algorithms.mappo import CoopPPOLearner
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.teachers.interfaces import TeacherResult
from vlm4coop.envs.lbf.prompts import joint_prompt
from vlm4coop.envs.lbf.prompts import SYSTEM_PROMPT
from vlm4coop.teachers.api import OpenCodeTeacher, build_api_cache_identity, api_joint_prompt, api_system_prompt
from vlm4coop.envs.scenario_sampling import TrainingSeedPool


class Log:
    def log_stat(self,*args): pass


def collect(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    arrays={}; meta={}; config=load_config(); torch.set_num_threads(2)
    meta['args']={str(n):vars(make_args(config,n,0)) for n in (3,4,5)}
    sampler=TrainingSeedPool(config)
    meta['seed_draws']=[sampler.sample() for _ in range(5)]
    meta['prompts']={str(n):[SYSTEM_PROMPT,joint_prompt(n),api_system_prompt(n),api_joint_prompt(n)] for n in (3,4,5)}
    first_rgb=None
    for n in (3,4,5):
        key=register_lbf(n,config['environment'])
        env=CoopGymmaWrapper(key=key,time_limit=100,pretrained_wrapper=None,seed=0,common_reward=True,reward_scalarisation='sum')
        try:
            for seed in (7,19):
                env.reset(seed=seed)
                rng=np.random.Generator(np.random.PCG64(seed+n))
                for t in range(8):
                    prefix=f'env_{n}_{seed}_{t}'
                    arrays[prefix+'_obs']=np.asarray(env.get_obs()).copy()
                    arrays[prefix+'_state']=np.asarray(env.get_state()).copy()
                    arrays[prefix+'_avail']=np.asarray(env.get_avail_actions()).copy()
                    arrays[prefix+'_rgb']=env.render().copy()
                    if first_rgb is None: first_rgb=env.render().copy()
                    actions=rng.integers(0,6,size=n)
                    arrays[prefix+'_actions']=actions
                    _,reward,done,trunc,info=env.step(actions)
                    meta[prefix]=dict(reward=reward,done=bool(done),truncated=bool(trunc),info=info)
        finally: env.close()
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        cache=ProbabilityCache(Path(tmp)/'isolated.sqlite',build_api_cache_identity(config['teacher']))
        image=Image.fromarray(first_rgb)
        meta['cache_key']=cache.key(image,3)
        texts=[json.dumps({'agents':[{'id':i,'probs':[2,-1,.5,.5,0,0]} for i in range(3)]})]
        payloads=[]
        class Response:
            status,headers=200,{}
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self):
                return json.dumps({'id':'offline','choices':[{'message':{'content':texts[0],'reasoning_content':'planning'}}],
                    'usage':{'completion_tokens_details':{'reasoning_tokens':10}}}).encode()
        def fake_open(request,**unused): payloads.append(json.loads(request.data)); return Response()
        with patch.dict(os.environ,{'OPENCODE_API_KEY':'WITNESS-TEST-ONLY'}),patch('vlm4coop.teachers.api.urlopen',fake_open):
            teacher=OpenCodeTeacher(config['teacher'],Path(tmp)/'calls.jsonl',cache)
            result=teacher.predict(image,3); again=teacher.predict(image,3)
            arrays['api_probs']=result.probs
            meta['api']=dict(calls=teacher.calls,cache_hit=again.cache_hit,repaired=result.repaired,payloads=payloads)
        cache.close()
        for mode in ('mappo','soft','hard'):
            random.seed(23); np.random.seed(23); torch.manual_seed(23)
            args=make_args(config,3,0,'smoke'); args.batch_size_run=2
            args.kd_mode='soft' if mode=='mappo' else mode; args.kd_lambda_current=0. if mode=='mappo' else .1
            runner=eval_runner=None
            try:
                runner,mac,scheme=setup_runner(args,Log(),config)
                runner.next_episode_seeds=[7,19]; runner.capture_frames=True
                with torch.no_grad(): batch=runner.run(test_mode=False)
                meta[mode+'_episodes']=runner.last_episode_records
                for (worker,t),rgb in runner.frames.items():
                    if t in (0,99): arrays[f'{mode}_frame_{worker}_{t}']=rgb
                class Synthetic:
                    def predict(self,image,n_agents):
                        return TeacherResult(np.tile([.1,.1,.1,.1,.1,.5],(n_agents,1)).astype(np.float32),'SYNTHETIC WITNESS',False)
                annotate(runner,batch,Synthetic())
                batch=batch[:,:int(batch.max_t_filled())]
                for name,value in batch.data.transition_data.items(): arrays[mode+'_batch_'+name]=value.numpy().copy()
                learner=(PPOLearner if mode=='mappo' else CoopPPOLearner)(mac,scheme,Log(),args)
                learner.train(batch,runner.t_env,0)
                for component,module in [('actor',learner.mac.agent),('critic',learner.critic),('target',learner.target_critic),('old_actor',learner.old_mac.agent)]:
                    for name,value in module.state_dict().items(): arrays[f'{mode}_{component}_{name}']=value.cpu().numpy().copy()
                for name in ('rew_ms','ret_ms'):
                    if hasattr(learner,name):
                        for attr in ('mean','var','count'):
                            arrays[f'{mode}_{name}_{attr}']=np.asarray(getattr(getattr(learner,name),attr)).copy()
                eval_args=deepcopy(args); eval_runner,_,_=setup_runner(eval_args,Log(),config,mac)
                local=deepcopy(config); local['eval_episodes_every_checkpoint']=2
                destination=Path(tmp)/mode; destination.mkdir()
                rng_before=torch.get_rng_state().clone()
                meta[mode+'_evaluation']=_json_safe(evaluate(eval_runner,local,destination,1000,runner.t_env,0))
                assert torch.equal(rng_before,torch.get_rng_state())
                meta[mode+'_eval_episodes']=[json.loads(row) for row in (destination/'eval_episodes.jsonl').read_text().splitlines()]
            finally:
                for instance in (runner,eval_runner):
                    if instance is not None: instance.close_env()
    np.savez_compressed(output/'arrays.npz',**arrays)
    (output/'metadata.json').write_text(json.dumps(_json_safe(meta),sort_keys=True,indent=2,allow_nan=False))
    print(json.dumps(dict(status='collected',arrays=len(arrays),metadata_fields=len(meta),real_api_calls=0)))


def compare(before,after):
    before,after=Path(before),Path(after)
    a,b=np.load(before/'arrays.npz'),np.load(after/'arrays.npz')
    assert set(a.files)==set(b.files)
    for name in a.files: np.testing.assert_array_equal(a[name],b[name],err_msg=name)
    assert json.loads((before/'metadata.json').read_text())==json.loads((after/'metadata.json').read_text())
    print(json.dumps(dict(status='exact_parity',arrays=len(a.files),metadata_equal=True,real_api_calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output'); p.add_argument('--compare-to')
    opts=p.parse_args()
    if opts.compare_to: compare(opts.compare_to,opts.output)
    else: collect(opts.output)
