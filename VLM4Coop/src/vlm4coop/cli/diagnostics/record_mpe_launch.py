"""Verify a new MPE launch by frozen metadata only; never read live scores."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from vlm4coop.utils.paths import PROJECT
from vlm4coop.training.configuration import make_args
from vlm4coop.envs.scenario_sampling import TrainingSeedPool


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path)
    p.add_argument('--container',required=True); p.add_argument('--session',required=True,type=int)
    a=p.parse_args(); run=a.run.resolve()
    if not run.is_relative_to(PROJECT.resolve()): raise ValueError('Project-owned output required')
    frozen=json.loads((run/'config.json').read_text()); c=frozen['experiment_config']
    assert frozen['method']=='mappo' and frozen['profile']=='formal' and frozen['agents']==3 and frozen['seed']==0
    assert c['environment_adapter']=='mpe_simple_spread' and c['environment']['max_episode_steps']==25
    assert c['formal_train_steps']==1000000 and c['batch_size_run']==c['batch_size']==c['buffer_size']==40 and c['epochs']==4
    assert c['eval_interval_steps']==1000 and frozen['epymarl_args']['test_interval']==1000
    assert c['fixed_eval_seeds']==dict(start=1000000,count=100) and c['eval_episodes_every_checkpoint']==100
    assert c['eval_parallel_envs']==10 and c['test_greedy'] is False
    assert frozen['epymarl_args']==vars(make_args(c,3,0,'formal'))
    sampler=TrainingSeedPool(c)
    assert sampler.seeds==list(range(500)) and sampler.metadata==json.loads((run/'training_seed_pool.json').read_text())
    protocol=PROJECT/c['protocol_document']
    assert frozen['protocol_source_sha256']==sha256(protocol.read_bytes()).hexdigest()
    provenance=json.loads((run/'provenance.json').read_text())
    assert provenance['packages']['pettingzoo']=='1.24.3'
    assert provenance['files'][c['protocol_document']]==frozen['protocol_source_sha256']
    assert not any(name.startswith('src/old/') for name in provenance['files'])
    baseline=json.loads((PROJECT/'results/mappo_8x8_formal_fixed500_1000k_20261017_01/config.json').read_text())['epymarl_args']
    actual=frozen['epymarl_args']; differences=[key for key in actual if actual[key]!=baseline.get(key)]
    assert sorted(differences)==['batch_size','batch_size_run','buffer_size','env','env_args','n_actions']
    previous=json.loads((PROJECT/'results/mappo_mpe_spread_fixed500_1000k_20261020_01/config.json').read_text())['epymarl_args']
    previous_differences=[key for key in actual if actual[key]!=previous.get(key)]
    assert sorted(previous_differences)==['batch_size','batch_size_run','buffer_size']
    for name in ('start_mpe_mappo_1000k.sh','mpe_mappo_train_and_finish.sh','docker_mpe.sh'):
        copyfile(PROJECT/'scripts'/name,run/name)
    result=dict(status='launch_verified',method='mappo',environment='mpe_simple_spread',agents=3,landmarks=3,
        train_steps_budget=1000000,parallel_envs=40,episode_steps=25,training_seed_pool=500,test_seed_pool=100,
        native_PPO_epochs=4,student_arg_differences_vs_LBF=sorted(differences),
        student_arg_differences_vs_previous_MPE=previous_differences,
        update_interval_steps=1000,evaluation_interval_steps=1000,expected_evaluations=1000,
        source_and_protocol_frozen=True,teacher_used=False,intermediate_scores_read=False,new_API_requests=0)
    (run/'launch_validation.json').write_text(json.dumps(result,indent=2))
    state=PROJECT/'reports/PIPELINE_STATE.json'; s=json.loads(state.read_text())
    prior_mpe=s.get('mpe',{})
    s['mpe']=dict(status='mappo_running',run=str(run.relative_to(PROJECT)),container=a.container,
        exec_session=a.session,methods_implemented=['mappo','pure_vlm','soft','hard'],
        teacher_formal_started=prior_mpe.get('teacher_formal_started',False),train_steps=1000000,agents=3,landmarks=3,episode_steps=25,
        parallel_envs=40,train_scenarios=500,eval_scenarios=100,
        update_interval_steps=1000,evaluation_interval_steps=1000,expected_evaluations=1000,
        automatic_figures=['learning_curves.png','learning_curves.pdf','learning_curves.svg'],
        protocol=c['protocol_document'],monitoring='one_launch_check_then_wait_no_frequent_scores')
    for key in ('pure_vlm_formal','current_teacher_probe','last_teacher_probe','teacher_control_validation'):
        if key in prior_mpe:
            s['mpe'][key]=prior_mpe[key]
    s['mpe']['active_protocol']=c['protocol_document']
    s['mpe']['formal_standard']=dict(parallel_envs=40,episode_steps=25,
        update_interval_steps=1000,PPO_epochs=4,evaluation_interval_steps=1000)
    s.update(formal_experiments='mpe_mappo_running',formal_run=str(run.relative_to(PROJECT)),
             formal_container=a.container,formal_exec_session=a.session,
             current_tests='See '+str(Path(c['protocol_document']).parent/'VALIDATION.md'),
             monitoring='verify_launch_once_wait_natural_exit_then_analyze',updated_at=datetime.now(timezone.utc).isoformat())
    state.write_text(json.dumps(s,indent=2)); print(json.dumps(result,indent=2))


def record_user_stop(path, reason):
    """Record the user's stop after the project container has actually exited."""
    run=Path(path).resolve()
    if not run.is_relative_to(PROJECT.resolve()): raise ValueError('Project output required')
    prior_bytes=(run/'status.json').read_bytes()
    prior=json.loads(prior_bytes)
    if prior['status']=='complete': raise ValueError('Do not mark a completed run as interrupted')
    # Retain the exact final status bytes before explicitly marking interruption.
    with (run/'status_before_user_stop.json').open('xb') as stream: stream.write(prior_bytes)
    now=datetime.now(timezone.utc).isoformat()
    batches=[json.loads(line) for line in (run/'train_batches.jsonl').read_text().splitlines() if line.strip()]
    sampled=batches[-1]['actual_train_env_steps'] if batches else 0
    checkpoints=sorted(int(p.name) for p in (run/'checkpoints').iterdir() if p.is_dir())
    record=dict(status='stopped_by_user',stopped_at=now,reason=reason,
        last_reported_completed_update_env_steps=prior['train_env_steps'],
        last_recorded_sampled_env_steps=sampled,retained_checkpoint_steps=checkpoints,
        latest_checkpoint_steps=max(checkpoints,default=None),
        checkpoint_at_stop_created=False,exact_resume_supported=False,auto_restart=False,
        raw_logs_frozen_config_source_and_checkpoints_preserved=True,
        note='Status confirms completed updates only through its recorded counter; the last sampled batch may have been interrupted during update or evaluation.')
    with (run/'interruption.json').open('x') as stream: json.dump(record,stream,indent=2)
    prior.update(status='stopped_by_user',stopped_at=now,stop_reason=reason,
        sampled_env_steps=sampled,auto_restart=False,phase='stopped_by_user')
    (run/'status.json').write_text(json.dumps(prior,indent=2))
    state=PROJECT/'reports/PIPELINE_STATE.json'; s=json.loads(state.read_text())
    relative=str(run.relative_to(PROJECT))
    if s.get('mpe',{}).get('run')==relative:
        s['mpe'].update(status='paused_by_user',run_status='stopped_by_user',container=None,
            exec_session=None,auto_restart=False,interruption=relative+'/interruption.json',
            active_protocol='reports/mpe_restore_eval1000_20261020/PROTOCOL.md',
            formal_standard=dict(update_interval_steps=250,PPO_epochs=4,evaluation_interval_steps=1000))
        s.update(formal_experiments='mpe_paused_by_user',formal_container=None,formal_exec_session=None,
            auto_restart=False,user_pause_reason=reason,updated_at=now)
        state.write_text(json.dumps(s,indent=2))
    print(json.dumps(record,indent=2))
    return record


def record_completion(path):
    run=Path(path).resolve()
    if not run.is_relative_to(PROJECT.resolve()): raise ValueError('Project output required')
    status=json.loads((run/'status.json').read_text())
    analysis=Path(str(run)+'_analysis')
    assert status['status']=='complete' and json.loads((analysis/'validation.json').read_text())['status']=='passed'
    for extension in ('png','pdf','svg'):
        assert (analysis/f'learning_curves.{extension}').stat().st_size>0
    state=PROJECT/'reports/PIPELINE_STATE.json'; s=json.loads(state.read_text())
    relative=str(run.relative_to(PROJECT))
    if s.get('mpe',{}).get('run')==relative:
        s['mpe'].update(status='mappo_complete_validated',container=None,exec_session=None,
            actual_train_env_steps=status['train_env_steps'],episodes=status['episodes'],
            analysis=str(analysis.relative_to(PROJECT)))
        if s.get('formal_run')==relative:
            s.update(formal_experiments='mpe_mappo_complete_teacher_runs_not_started',
                formal_container=None,formal_exec_session=None)
        s['updated_at']=datetime.now(timezone.utc).isoformat()
        state.write_text(json.dumps(s,indent=2))


if __name__=='__main__': main()
