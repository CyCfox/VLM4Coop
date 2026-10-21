"""Real original-MPE dynamics/native PPO; teacher HTTP is always mocked."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
pytest.importorskip('pettingzoo',reason='MPE tests require the isolated docker_mpe runtime')
import torch
from learners.ppo_learner import PPOLearner
from vlm4coop.training.configuration import load_config,make_args
from vlm4coop.utils.paths import PROJECT
from vlm4coop.envs.mpe.adapter import MPEGymBridge
from vlm4coop.envs.mpe.rendering import render_geometry,global_rgb,SIZE,AGENT_COLORS
from vlm4coop.envs.registry import get_adapter
from vlm4coop.training.setup import setup_runner
from vlm4coop.training.annotation import annotate
from vlm4coop.algorithms.mappo import CoopPPOLearner
from vlm4coop.teachers.interfaces import TeacherResult
from vlm4coop.teachers.api import OpenCodeTeacher,build_api_cache_identity
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.evaluation.pure_vlm import TeacherMAC
from vlm4coop.evaluation.metrics import summarize,checkpoint_rank


def cfg(): return load_config(PROJECT/'configs/mpe_spread.yaml')


@pytest.mark.parametrize('action,velocity',[(0,[0,0]),(1,[-.5,0]),(2,[.5,0]),(3,[0,-.5]),(4,[0,.5])])
def test_native_action_order_is_force_not_instant_move(action,velocity):
    env=MPEGymBridge()
    try:
        env.reset(seed=7)
        for i,a in enumerate(env.world.agents): a.state.p_pos=np.array([float(i*3),0.]); a.state.p_vel=np.zeros(2)
        before=env.world.agents[0].state.p_pos.copy()
        env.step([action,0,0])
        np.testing.assert_array_equal(env.world.agents[0].state.p_pos,before)
        np.testing.assert_allclose(env.world.agents[0].state.p_vel,velocity,atol=1e-12)
    finally: env.close()


def test_none_inertia_and_velocity_rendering_are_true_state():
    env=MPEGymBridge()
    try:
        env.reset(seed=7)
        for i,a in enumerate(env.world.agents): a.state.p_pos=np.array([float(i*3),0.]); a.state.p_vel=np.zeros(2)
        a=env.world.agents[0]; a.state.p_vel=np.array([.4,.3]); before=a.state.p_pos.copy()
        g=render_geometry(env.world); item=g['agents'][0]
        assert item['label']=='V=0.50'
        assert item['tip'][0]>item['start'][0] and item['tip'][1]<item['start'][1]
        np.testing.assert_allclose(np.asarray(item['tip'])-item['start'],np.array([.8,-.6])*.5*item['radius'])
        first=global_rgb(env.world); np.testing.assert_array_equal(first,global_rgb(env.world))
        assert first.shape==(g['image_size'][1],g['image_size'][0],3) and first.dtype==np.uint8
        assert max(first.shape[:2])==SIZE
        env.step([0,0,0])
        np.testing.assert_allclose(a.state.p_pos,before+np.array([.4,.3])*.1)
        np.testing.assert_allclose(a.state.p_vel,[.3,.225])
    finally: env.close()


def test_bridge_preserves_original_rewards_states_final_observation_and_silent_inputs():
    from pettingzoo.mpe import simple_spread_v3
    native=simple_spread_v3.parallel_env(N=3,local_ratio=.5,max_cycles=25,continuous_actions=False)
    bridge=MPEGymBridge()
    try:
        obs,_=bridge.reset(seed=19); original,_=native.reset(seed=19)
        for i,name in enumerate(bridge.agent_order): np.testing.assert_array_equal(obs[i],original[name][:-4])
        actions=np.random.Generator(np.random.PCG64(11)).integers(0,5,size=(25,3))
        for t,row in enumerate(actions):
            obs,reward,done,trunc,info=bridge.step(row)
            native_obs,native_reward,_,native_trunc,_=native.step(dict(zip(bridge.agent_order,map(int,row))))
            for i,name in enumerate(bridge.agent_order):
                np.testing.assert_array_equal(obs[i],native_obs[name][:-4]); assert reward[i]==native_reward[name]
            np.testing.assert_array_equal([a.state.p_pos for a in bridge.world.agents],[a.state.p_pos for a in native.unwrapped.world.agents])
            np.testing.assert_array_equal([a.state.p_vel for a in bridge.world.agents],[a.state.p_vel for a in native.unwrapped.world.agents])
            assert not done and trunc==(t==24) and info['episode_limit']==(t==24)
        assert len(native.agents)==0 and all(x.shape==(14,) for x in obs)
        saved=bridge.render().copy(); bridge.reset(seed=19); reset=bridge.render().copy()
        bridge.step([1,2,3]); bridge.reset(seed=19); np.testing.assert_array_equal(reset,bridge.render())
        assert not np.array_equal(saved,reset)
    finally: native.close(); bridge.close()


def test_all_600_fixed_scenarios_reset_reproducibly_and_disjoint():
    env=MPEGymBridge()
    try:
        signatures=set()
        for seed in list(range(500))+list(range(1000000,1000100)):
            first,_=env.reset(seed=seed)
            positions=np.asarray([a.state.p_pos for a in env.world.entities]).copy()
            signatures.add(positions.tobytes())
            second,_=env.reset(seed=seed)
            np.testing.assert_array_equal(first,second)
            assert all(np.array_equal(a.state.p_vel,np.zeros(2)) for a in env.world.agents)
        assert len(signatures)==600
    finally: env.close()


@pytest.mark.parametrize('mode',['soft','hard'])
def test_real_mpe_four_epoch_distillation_and_failed_masks_match_native_ppo(mode):
    config=cfg(); config.update(batch_size_run=10,batch_size=10,buffer_size=10)  # Historical 250-step regression fixture.
    args=make_args(config,3,0,'smoke'); args.kd_mode=mode; args.kd_lambda_current=.1
    logger=SimpleNamespace(log_stat=lambda *unused:None); torch.manual_seed(13); torch.set_num_threads(2)
    runner,mac,scheme=setup_runner(args,logger,config)
    try:
        runner.next_episode_seeds=list(range(10)); runner.capture_frames=True
        with torch.no_grad(): batch=runner.run(test_mode=False)
        assert runner.t_env==250 and batch['obs'].shape==(10,26,3,14)
        assert batch['teacher_probs'].shape==(10,26,3,5) and not batch['terminated'].any()
        assert len(runner.frames)==250 and all(r['length']==25 and r['episode_limit'] for r in runner.last_episode_records)
        original_reward=batch['reward'].clone()
        class Teacher:
            calls=0
            def predict(self,image,n_agents):
                self.calls+=1
                return TeacherResult(np.tile([.05,.05,.8,.05,.05],(3,1)).astype(np.float32),'SYNTHETIC MPE LABEL',False)
        teacher=Teacher(); annotate(runner,batch,teacher)
        assert teacher.calls==250 and not runner.frames and torch.equal(original_reward,batch['reward'])
        assert int(batch['teacher_valid'].sum())==250
        batch=batch[:,:int(batch.max_t_filled())]
        baseline=PPOLearner(deepcopy(mac),scheme,logger,deepcopy(args))
        distilled=CoopPPOLearner(deepcopy(mac),scheme,logger,deepcopy(args))
        masked=CoopPPOLearner(deepcopy(mac),scheme,logger,deepcopy(args))
        for item in (distilled,masked):
            item.critic.load_state_dict(baseline.critic.state_dict()); item.target_critic.load_state_dict(baseline.target_critic.state_dict())
        initial=[p.detach().clone() for p in mac.parameters()]
        baseline.train(batch,250,0); distilled.train(batch,250,0)
        batch['teacher_valid'].zero_(); masked.train(batch,250,0)
        assert any(not torch.equal(x,y) for x,y in zip(initial,distilled.mac.parameters()))
        assert any((x-y).abs().max()>1e-8 for x,y in zip(baseline.mac.parameters(),distilled.mac.parameters()))
        for x,y in zip(baseline.mac.parameters(),masked.mac.parameters()): torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        for x,y in zip(baseline.critic.parameters(),masked.critic.parameters()): torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        assert args.epochs==4 and teacher.calls==250
    finally: runner.close_env()


def test_mpe_teacher_five_actions_velocity_prompt_and_cache_isolation(monkeypatch,tmp_path):
    config=cfg(); spec=get_adapter(config).teacher_spec()
    assert spec.actions==('NONE','LEFT','RIGHT','DOWN','UP')
    assert '0.75' in spec.user_prompt(3) and 'velocity' in spec.user_prompt(3)
    identity=build_api_cache_identity(config['teacher'],spec)
    lbf=build_api_cache_identity(load_config()['teacher'])
    cache=ProbabilityCache(tmp_path/'mpe.sqlite',identity); old=ProbabilityCache(tmp_path/'lbf.sqlite',lbf)
    image=Image.new('RGB',(720,720))
    assert cache.key(image,3)!=old.key(image,3)
    monkeypatch.setenv('OPENCODE_API_KEY','MPE-TEST-ONLY')
    requests=[]
    class Reply:
        status,headers=200,{}
        def __enter__(self): return self
        def __exit__(self,*unused): pass
        def read(self):
            return json.dumps(dict(choices=[dict(message=dict(content='Reasoning. '+json.dumps(dict(agents=[
                dict(id=i,probs=[2,-1,.5,.5,0]) for i in range(3)])),reasoning_content='extra reasoning' if len(requests)>3 else ''))])).encode()
    def open_fake(request,**unused):
        payload=json.loads(request.data)
        assert payload['messages'][0]['content']==spec.system_prompt(3)
        assert '0=NONE' in payload['messages'][1]['content'][0]['text']
        assert len(payload['messages'][1]['content'])==2
        assert payload['thinking']=={'type':'disabled'}
        assert payload['chat_template_kwargs']==dict(thinking=False,enable_thinking=False)
        assert 'reasoning_effort' not in payload
        requests.append(payload)
        return Reply()
    monkeypatch.setattr('vlm4coop.teachers.api.urlopen',open_fake)
    try:
        teacher=OpenCodeTeacher(config['teacher'],tmp_path/'calls.jsonl',cache,protocol=spec)
        for _ in range(3):
            teacher.request(spec.user_prompt(3),image,system=spec.system_prompt(3))
        result=teacher.predict(image,3); assert result.repaired and result.probs.shape==(3,5)
        np.testing.assert_allclose(result.probs[0],[.5,0,.25,.25,0])
        assert teacher.predict(image,3).cache_hit and teacher.calls==4
    finally: cache.close(); old.close()


def test_mpe_pure_controller_uncached_full_100_episode_path(monkeypatch,tmp_path):
    from vlm4coop.evaluation import pure_vlm
    config=cfg(); monkeypatch.setattr(pure_vlm,'PROJECT',tmp_path)
    spec=get_adapter(config).teacher_spec()
    class Synthetic:
        config={'pure_vlm_use_cache':False}; protocol=spec; cache=None
        calls=0
        def predict_many(self,images,n_agents,use_cache=True):
            assert not use_cache
            self.calls+=len(images)
            return [TeacherResult(np.tile([1,0,0,0,0],(3,1)).astype(np.float32),'SYNTHETIC PURE TEST',False)]*len(images)
    teacher=Synthetic(); monkeypatch.setattr(pure_vlm,'make_teacher',lambda *args,**kwargs:teacher)
    result=pure_vlm.run(tmp_path/'pure',config=config,profile='smoke')
    assert result['n_eval']==100 and result['decisions']==2500 and teacher.calls==2500
    assert result['parallel_envs']==10 and result['cache_hits']==0
    assert len(list((tmp_path/'pure/frames').glob('*.png')))==2500
    episodes=[json.loads(row) for row in (tmp_path/'pure/eval_episodes.jsonl').read_text().splitlines()]
    assert [r['env_seed'] for r in episodes]==list(range(1000000,1000100))
    assert all(r['length']==25 and r['environment']=='mpe_simple_spread' for r in episodes)
    summary=summarize(episodes)
    assert math.isnan(summary['success_episodes_length_mean']) and 'final_coverage_fraction_mean' in summary
    assert checkpoint_rank(summary)[0]==summary['return_mean']


def test_agent_body_occludes_target_fill_and_name_without_removing_target():
    env=MPEGymBridge()
    try:
        env.reset(seed=7)
        env.world.landmarks[0].state.p_pos=env.world.agents[0].state.p_pos.copy()
        image=env.render(); g=render_geometry(env.world)
        x,y=g['goals'][0]['center']; r=g['goals'][0]['radius']
        np.testing.assert_array_equal(image[int(y-.7*r),int(x-.2*r)],[216,75,66])
        # Moving a fully occluded target within the agent must not expose its name.
        env.world.landmarks[0].state.p_pos+=np.array([.02,0])
        np.testing.assert_array_equal(image,env.render())
        assert len(env.world.landmarks)==3
        a=g['agents'][0]
        assert a['speed']==0 and a['label']=='V=0.00' and a['start']==a['tip']
        assert a['arrow_polygon']==[]
    finally: env.close()


@pytest.mark.parametrize('velocity',[[.1,0],[0,.1],[-.1,0],[0,-.1]])
def test_covered_landmark_does_not_hide_low_speed_vector(velocity):
    env=MPEGymBridge()
    try:
        env.reset(seed=7)
        # Isolate the covered agent so no unrelated marker/label intersects it.
        for entity,position in zip(env.world.entities,
                [[0,0],[-.8,-.8],[.8,.8],[0,0],[-.8,.8],[.8,-.8]]):
            entity.state.p_pos=np.asarray(position,dtype=float)
        env.world.agents[0].state.p_vel=np.asarray(velocity)
        image=env.render(); g=render_geometry(env.world); agent=g['agents'][0]
        start=np.asarray(agent['start']); tip=np.asarray(agent['tip'])
        assert np.linalg.norm(tip-start)==pytest.approx(.5*agent['radius'])
        visible=np.rint(start+.84*(tip-start)).astype(int)
        np.testing.assert_array_equal(image[visible[1],visible[0]],[255,255,255])
        # The agent hides the entire smaller target; its own arrow stays visible.
        x,y=g['goals'][0]['center']; r=g['goals'][0]['radius']
        patch=image[int(y-r+3):int(y+r-2),int(x-r+3):int(x+r-2)]
        assert not np.any(np.all(patch==[76,153,85],axis=-1))
    finally: env.close()


def test_higher_agent_id_body_is_drawn_over_lower_id():
    from PIL import ImageColor
    env=MPEGymBridge()
    try:
        env.reset(seed=7)
        for agent in env.world.agents: agent.state.p_pos=np.zeros(2)
        image=env.render(); item=render_geometry(env.world)['agents'][2]
        x,y=item['start']; r=item['radius']
        np.testing.assert_array_equal(image[int(y),int(x+.7*r)],ImageColor.getrgb(AGENT_COLORS[2]))
    finally: env.close()


def test_internal_arrow_fixed_length_independent_of_speed_and_viewport():
    env=MPEGymBridge()
    try:
        env.reset(seed=7)
        first=None
        for velocity in ([.4,.3],[4.,3.],[.04,.03]):
            env.world.agents[0].state.p_vel=np.asarray(velocity)
            g=render_geometry(env.world); a=g['agents'][0]
            distances=np.linalg.norm(np.asarray(a['arrow_polygon'])-a['start'],axis=1)
            assert distances.max()<=.5*a['radius']+1e-10
            assert a['label']==f'V={np.linalg.norm(velocity):.2f}'
            if first is not None:
                assert g['extent']==first['extent'] and g['scale']==first['scale']
                np.testing.assert_allclose(a['arrow_polygon'],first['agents'][0]['arrow_polygon'])
                np.testing.assert_allclose(a['tip'],first['agents'][0]['tip'])
            first=g
    finally: env.close()


def test_internal_labels_circle_targets_and_no_outer_text(monkeypatch):
    from PIL import ImageDraw
    env=MPEGymBridge(); labels=[]
    original=ImageDraw.ImageDraw.text
    def observed(draw,xy,text,*args,**kwargs):
        labels.append((xy,text,kwargs))
        return original(draw,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',observed)
    try:
        env.reset(seed=7)
        for entity,position in zip(env.world.entities,
                [[-.65,-.65],[0,-.65],[-.65,0],[.65,.65],[.65,-.65],[-.65,.65]]):
            entity.state.p_pos=np.asarray(position,dtype=float)
        image=env.render(); g=render_geometry(env.world)
        names=[label for _,label,_ in labels]
        assert sorted(label for label in names if not label.startswith('V='))==['A0','A1','A2','G0','G1','G2']
        assert len(names)==9 and names.count('V=0.00')==3
        for xy,label,kwargs in labels:
            box=ImageDraw.Draw(Image.fromarray(image)).textbbox(xy,label,font=kwargs['font'],
                stroke_width=kwargs.get('stroke_width',0))
            if label.startswith('G'):
                assert kwargs.get('stroke_width',0)==0 and 'stroke_fill' not in kwargs
                item=g['goals'][int(label[1:])]; center=np.asarray(item['center'])
            elif label.startswith('A'):
                item=g['agents'][int(label[1:])]; center=np.asarray(item['start'])
            else:
                item=min(g['agents'],key=lambda a:np.linalg.norm(np.asarray(a['speed_center'])-xy))
                center=np.asarray(item['start'])
            for corner in ((box[0],box[1]),(box[2],box[3]),(box[0],box[3]),(box[2],box[1])):
                assert np.linalg.norm(np.asarray(corner)-center)<=item['radius']+1
        goal=g['goals'][0]; x,y=goal['center']; r=goal['radius']
        np.testing.assert_array_equal(image[int(y+.8*r),int(x+.8*r)],[255,255,255])
        np.testing.assert_array_equal(image[int(y-.7*r),int(x-.2*r)],[76,153,85])
        from PIL import ImageColor
        for a in g['agents']:
            x,y=a['start']; r=a['radius']
            np.testing.assert_array_equal(image[int(y),int(x+.7*r)],ImageColor.getrgb(AGENT_COLORS[a['id']]))
        assert g['scale']>350  # No old 60px framing/header margin.
        entities=[(a['start'],a['radius']) for a in g['agents']]+[(g['center'],g['radius']) for g in g['goals']]
        lower=np.min([np.asarray(p)-r for p,r in entities],axis=0)
        upper=np.max([np.asarray(p)+r for p,r in entities],axis=0)
        assert g['image_size']==[1440,1440]
        assert np.all(lower>=1-1e-8) and np.all(upper<=1439+1e-8)
        np.testing.assert_allclose(lower+upper,[1440,1440])
        assert min(lower)==pytest.approx(1)
    finally: env.close()


def test_downward_arrow_does_not_overlap_own_name(monkeypatch):
    from PIL import ImageDraw
    labels={}; original=ImageDraw.ImageDraw.text
    def observed(draw,xy,text,*args,**kwargs):
        labels[text]=draw.textbbox(xy,text,font=kwargs['font'])
        return original(draw,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',observed)
    env=MPEGymBridge()
    try:
        env.reset(seed=7)
        for _ in range(5): env.step([1,2,3])
        image=env.render(); a=render_geometry(env.world)['agents'][2]
        assert env.world.agents[2].state.p_vel[1]<0
        assert a['tip'][1]>a['start'][1]
        assert max(p[1] for p in a['arrow_polygon'])<labels['A2'][1]
        x,y=a['tip']
        assert image.shape[0]>y and x>=0
    finally: env.close()


def test_pure_generic_validation_has_zero_training_steps(monkeypatch,tmp_path):
    import runpy
    import sys
    from vlm4coop.utils.json_logging import _json_safe
    config=cfg(); config['eval_episodes_every_checkpoint']=2
    env=MPEGymBridge(); adapter=get_adapter(config); episodes=[]
    try:
        for seed in range(1000000,1000002):
            env.reset(seed=seed); total=0
            for _ in range(25):
                _,reward,_,_,info=env.step([0,0,0]); total+=sum(reward)
            episodes.append(dict(adapter.episode_fields(info),worker=0,env_seed=seed,eval_seed=seed,
                length=25,team_return=total,scheduled_eval_step=0,actual_train_env_steps=0))
    finally: env.close()
    (tmp_path/'config.json').write_text(json.dumps(dict(experiment_config=config,method='pure_vlm',agents=3,
        profile='smoke',test_fixture='Real all-NONE environment transitions; not a VLM result')))
    (tmp_path/'status.json').write_text(json.dumps(dict(status='complete')))
    (tmp_path/'eval_episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    summary=summarize(episodes); summary.update(scheduled_eval_step=0,actual_train_env_steps=0)
    (tmp_path/'evaluations.jsonl').write_text(json.dumps(_json_safe(summary))+'\n')
    monkeypatch.setattr(sys,'argv',['validate_outputs',str(tmp_path)])
    runpy.run_module('vlm4coop.cli.validation.validate_outputs',run_name='__main__')


def test_mpe_frozen_protocol_is_in_source_provenance(tmp_path):
    from hashlib import sha256
    import tarfile
    from vlm4coop.utils.provenance import capture
    config=cfg(); document=PROJECT/config['protocol_document']
    (tmp_path/'config.json').write_text(json.dumps(dict(experiment_config=config)))
    capture(tmp_path)
    records=json.loads((tmp_path/'provenance.json').read_text())['files']
    assert records[config['protocol_document']]==sha256(document.read_bytes()).hexdigest()
    with tarfile.open(tmp_path/'source_snapshot.tar.gz') as archive:
        assert archive.extractfile('protocol/PROTOCOL.md').read()==document.read_bytes()
