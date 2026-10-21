"""Record launch configuration parity without reading intermediate scores."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import copyfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--container',required=True)
    parser.add_argument('--session',required=True,type=int)
    args=parser.parse_args()
    run=args.run
    frozen=json.loads((run/'config.json').read_text())
    soft=json.loads(Path('VLM4Coop/results/soft_8x8_fixed500_1000k_api_20261018_01/config.json').read_text())
    config=frozen['experiment_config']
    assert frozen['method']=='hard' and frozen['agents']==3 and frozen['seed']==0 and frozen['profile']=='formal'
    assert frozen['epymarl_args']==soft['epymarl_args']
    assert config['formal_train_steps']==1000000 and config['batch_size_run']==10 and config['epochs']==4
    assert config['training_seed_pool']==dict(start=0,count=500,sampler_seed=0,sampling='uniform_without_replacement_per_batch')
    assert config['teacher']['concurrency']==1000
    generation_fields=lambda teacher:{k:v for k,v in teacher.items() if k not in {'concurrency','pure_vlm_use_cache','timeout_seconds','parser_version','failure_policy','api_control_version'}}
    assert generation_fields(config['teacher'])==generation_fields(soft['experiment_config']['teacher'])
    policy=json.loads((run/'thinking_control_policy.json').read_text())
    assert policy['accept_hidden_reasoning'] is True and policy['max_attempts']==1 and policy['control_retries_enabled'] is False
    assert json.loads((run/'response_policy.json').read_text())['max_format_attempts']==3
    assert (run/'training_seed_pool.json').is_file()
    report=dict(status='launch_verified',method='hard',agents=3,seed=0,training_budget=1000000,
        fixed_training_scenarios=500,cache_first=True,http_concurrency_limit=1000,
        student_args_equal_soft=True,teacher_generation_config_equal_soft=True,
        returned_reasoning_accepted=True,thinking_control_retries=False,
        intermediate_scores_read=False,api_requests_during_check=0)
    (run/'launch_validation.json').write_text(json.dumps(report,indent=2))
    for source,target in [('start_hard_1000k.sh','start_script.sh'),('hard_train_and_finish.sh','train_and_finish_script.sh'),
        ('docker_api.sh','docker_script.sh'),('validation/validate_outputs.py','validate_script.py'),('analysis/export_training_run.py','export_script.py')]:
        copyfile(Path('VLM4Coop/scripts')/source,run/target)
    state_path=Path('VLM4Coop/reports/PIPELINE_STATE.json')
    state=json.loads(state_path.read_text())
    state['updated_at']=datetime.now(timezone.utc).isoformat()
    state['hard_previous_attempt']=state['hard_main']
    state['hard_main']=dict(status='running',run=str(run).removeprefix('VLM4Coop/'),cache_first=True,
        request_concurrency_limit=1000,agents=3,seed=0,budget=1000000,training_scenarios=500,
        container=args.container,exec_session=args.session,restart_from_initialization=True,
        preflight='results/hard_accept_reasoning_preflight_20261019_01',postprocess='same_container_validate_then_export')
    state.update(distillation_formal_training='hard_running_accept_valid_json_with_reasoning',
        formal_experiments='hard_running',formal_run=str(run).removeprefix('VLM4Coop/'),
        formal_container=args.container,formal_exec_session=args.session,
        current_tests='79_passed_63.12s_Docker',response_protocol='strict-schema-role-specific-v6',
        review='HARD_ACCEPT_REASONING_CODE_REVIEW_no_blocking_same_family_provisional',
        document_validation='HARD_ACCEPT_REASONING_DOC_VALIDATION_passed')
    state['thinking'].update(returned_reasoning='accept_if_JSON_valid_or_repairable',control_retries=0)
    state_path.write_text(json.dumps(state,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
