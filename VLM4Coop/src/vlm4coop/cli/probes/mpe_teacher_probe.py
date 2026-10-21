"""Three uncached RGB-only teacher-controlled real MPE episodes, exactly25steps."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from PIL import Image
from vlm4coop.envs.mpe.adapter import MPEGymBridge
from vlm4coop.envs.registry import get_adapter
from vlm4coop.evaluation.pure_vlm import TeacherMAC
from vlm4coop.evaluation.metrics import summarize
from vlm4coop.teachers.factory import make_teacher
from vlm4coop.teachers.api import build_api_cache_identity
from vlm4coop.training.configuration import load_config
from vlm4coop.utils.paths import PROJECT
from vlm4coop.utils.json_logging import append_json, _json_safe
from vlm4coop.utils.provenance import capture


def dump(path, value):
    Path(path).write_text(json.dumps(_json_safe(value),indent=2,ensure_ascii=False,allow_nan=False))


def world_state(env):
    return dict(agents=[dict(id=i,position=a.state.p_pos.tolist(),velocity=a.state.p_vel.tolist())
                        for i,a in enumerate(env.world.agents)],
        landmarks=[dict(id=i,position=g.state.p_pos.tolist()) for i,g in enumerate(env.world.landmarks)],
        metrics=env.state_metrics())


def run_probe(output, config=None, sample_only=False):
    config=deepcopy(config or load_config(PROJECT/'configs/mpe_spread.yaml'))
    if config.get('environment_adapter')!='mpe_simple_spread': raise ValueError('MPE only')
    config.update(eval_parallel_envs=3,pure_vlm_parallel_envs=3,pure_vlm_concurrency=3,
        eval_episodes_every_checkpoint=3)
    config['fixed_eval_seeds']['count']=3
    config['teacher'].update(concurrency=3,pure_vlm_use_cache=False)
    assert config['environment']['max_episode_steps']==25
    output=Path(output).resolve()
    if not output.is_relative_to((PROJECT/'results').resolve()): raise ValueError('Project results required')
    output.mkdir(parents=True,exist_ok=False)
    adapter=get_adapter(config); spec=adapter.teacher_spec()
    seeds=list(range(config['fixed_eval_seeds']['start'],config['fixed_eval_seeds']['start']+3))
    dump(output/'config.json',dict(experiment_config=config,method='pure_vlm',profile='probe',agents=3,
        formal=False,environment_seeds=seeds,system_prompt=spec.system_prompt(3),user_prompt=spec.user_prompt(3),
        input_mode='one_current_global_RGB_only',provided_current_coordinates=False,
        action_selection='Per-agent argmax; shared Pure NONE fallback after3format failures; no planner or action correction',
        note='3-episode capability probe, not the formal100scenario result'))
    dump(output/'cache_identity.json',build_api_cache_identity(config['teacher'],spec))
    capture(output)
    for folder in ('initial','final'): (output/folder).mkdir()
    envs=[]; teacher=None; mac=None; began=time.monotonic(); episode_records=[]
    returns=[0.,0.,0.]
    report=dict(status='initializing',formal=False,n_eval=3,parallel_envs=3,concurrency=3,
        horizon=25,cache_enabled=False,completed_joint_decisions=0,train_env_steps=0)
    dump(output/'status.json',report)
    try:
        for seed in seeds:
            env=MPEGymBridge(**config['environment']); envs.append(env); env.reset(seed=seed)
            dump(output/'initial'/f'seed{seed}.json',world_state(env))
            Image.fromarray(env.render()).save(output/'initial'/f'seed{seed}.png')
        if sample_only:
            report.update(status='sampled_no_api',teacher_calls=0)
        else:
            teacher=make_teacher(config,output,use_cache=False,purpose='pure')
            if teacher.cache is not None: raise ValueError('This pure-teacher probe must be uncached')
            mac=TeacherMAC(teacher,3,use_cache=False,output=output)
            mac.runner=SimpleNamespace(frames={},episode_seeds=seeds)
            report['status']='running'; dump(output/'status.json',report)
            for t in range(25):
                before=[world_state(env) for env in envs]
                mac.runner.frames={(i,t):env.render() for i,env in enumerate(envs)}
                actions=mac.select_actions(None,t,0,[0,1,2],True).tolist()
                for i,(env,action) in enumerate(zip(envs,actions)):
                    _,reward,terminated,truncated,info=env.step(action)
                    team_reward=float(sum(reward)); returns[i]+=team_reward
                    assert not terminated and bool(truncated)==(t==24)
                    append_json(output/'trajectory.jsonl',dict(worker=i,env_seed=seeds[i],t=t,
                        frame_file=f'worker{i:03d}_t{t:03d}.png',before=before[i],actions=action,
                        reward=team_reward,after=world_state(env),terminated=bool(terminated),truncated=bool(truncated),
                        episode_limit=bool(info['episode_limit'])))
                    if t==24:
                        row=dict(adapter.episode_fields(info),worker=i,env_seed=seeds[i],eval_seed=seeds[i],
                            train_seed=None,length=25,team_return=returns[i],scheduled_eval_step=0,actual_train_env_steps=0)
                        episode_records.append(row); append_json(output/'eval_episodes.jsonl',row)
                report['completed_joint_decisions']=mac.decisions
            assert mac.decisions==75 and len(episode_records)==3
            metrics=summarize(episode_records)
            metrics.update(scheduled_eval_step=0,actual_train_env_steps=0,train_seed=None)
            append_json(output/'evaluations.jsonl',metrics)
            report.update(status='complete',metrics=metrics,episodes=episode_records)
        for seed,env in zip(seeds,envs):
            Image.fromarray(env.render()).save(output/'final'/f'seed{seed}.png')
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),
            completed_joint_decisions=mac.decisions if mac else 0,
            partial_episode_returns=returns)
        raise
    finally:
        report['wall_seconds']=time.monotonic()-began
        report.update(teacher_calls=teacher.calls if teacher else 0,
            peak_active_http_requests=getattr(teacher,'max_active_requests',None),
            format_failures=getattr(teacher,'format_failures',0),format_retries=getattr(teacher,'format_retries',0),
            none_fallback_decisions=mac.none_fallbacks if mac else 0,
            responses_with_reasoning=getattr(teacher,'responses_with_reasoning',0),cache_hits=0,cache_writes=0)
        for env in envs: env.close()
        if teacher is not None and teacher.cache is not None: teacher.cache.close()
        dump(output/'status.json',report); dump(output/'summary.json',report)
        print(json.dumps(_json_safe(report),ensure_ascii=False,allow_nan=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-only',action='store_true',help='Three real resets/screenshots, no teacher/API')
    parser.add_argument('--protocol-document',help='Project-relative protocol to freeze for this authorized probe')
    args=parser.parse_args()
    config=load_config(PROJECT/'configs/mpe_spread.yaml')
    if args.protocol_document:
        config['protocol_document']=args.protocol_document
    run_probe(args.output,config=config,sample_only=args.sample_only)


if __name__=='__main__': main()
