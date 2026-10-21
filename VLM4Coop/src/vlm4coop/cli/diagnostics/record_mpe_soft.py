"""Metadata-only Soft launch verification and post-completion teacher accounting."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile

from vlm4coop.utils.paths import PROJECT
from vlm4coop.training.configuration import make_args
from vlm4coop.teachers.api import build_api_cache_identity
from vlm4coop.envs.registry import get_adapter


def read(path):
    return json.loads(path.read_text())


def verify_config(frozen):
    c = frozen['experiment_config']
    assert (frozen['method'], frozen['agents'], frozen['seed'], frozen['profile']) == ('soft', 3, 0, 'formal')
    assert c['formal_authorized_methods'] == ['soft']
    assert c['environment_adapter'] == 'mpe_simple_spread'
    assert c['formal_train_steps'] == 1_000_000 and c['environment']['max_episode_steps'] == 25
    assert c['batch_size_run'] == c['batch_size'] == c['buffer_size'] == 40 and c['epochs'] == 4
    assert c['eval_interval_steps'] == 1000 and c['eval_episodes_every_checkpoint'] == 100
    assert c['eval_parallel_envs'] == 10 and c['test_greedy'] is False
    assert c['training_seed_pool'] == dict(start=0, count=500, sampler_seed=0, sampling='uniform_without_replacement_per_batch')
    assert c['fixed_eval_seeds'] == dict(start=1000000, count=100)
    assert c['kd_supervision_stride'] == 1
    assert (c['kd_lambda_initial'], c['kd_lambda_plateau'], c['kd_anneal_t1'], c['kd_anneal_t2']) == (10., .1, 66000, 100000)
    t = c['teacher']
    assert t['backend'] == 'openai_api' and t['model'] == 'deepseek-v4.1-flash'
    assert t['endpoint'] == 'https://opencode.ai/inference/openai/v1/chat/completions'
    assert t['concurrency'] == t['reasoning_guard']['max_in_flight'] == 1000
    assert t['thinking'] == dict(type='disabled') and t['chat_template_kwargs'] == dict(thinking=False, enable_thinking=False)
    assert 'reasoning_effort' not in t and t['max_tokens'] is None and t['top_p'] == .95
    assert frozen['epymarl_args'] == vars(make_args(c, 3, 0, 'formal'))
    baseline = read(PROJECT/'results/mappo_mpe_spread_fixed500_1000k_batch40_20261020_03/config.json')
    assert frozen['epymarl_args'] == baseline['epymarl_args']
    identity = build_api_cache_identity(t, get_adapter(c).teacher_spec())
    assert identity == build_api_cache_identity(baseline['experiment_config']['teacher'], get_adapter(c).teacher_spec())
    return c, identity


def launch(run, container, session):
    frozen = read(run/'config.json'); c, identity = verify_config(frozen)
    protocol_hash = sha256((PROJECT/c['protocol_document']).read_bytes()).hexdigest()
    assert frozen['protocol_source_sha256'] == protocol_hash
    provenance = read(run/'provenance.json')
    assert provenance['files'][c['protocol_document']] == protocol_hash
    assert provenance['packages']['pettingzoo'] == '1.24.3'
    for name in ('start_mpe_soft_1000k.sh', 'mpe_soft_train_and_finish.sh', 'docker_mpe.sh', 'docker_mpe_api.sh'):
        copyfile(PROJECT/'scripts'/name, run/name)
    result = dict(status='launch_verified', method='soft', environment='mpe_simple_spread',
        training_steps_budget=1000000, parallel_envs=40, episode_steps=25, train_pool=500,
        test_pool=100, update_interval=1000, eval_interval=1000, PPO_epochs=4,
        teacher_cutoff_step=100000, last_annotated_update_step=99000,
        API_concurrency=1000, guard_max_in_flight=1000, cache_first=True,
        student_args_exact_MAPPO40_parity=True, generation_cache_identity_unchanged=True,
        cache_identity=identity, intermediate_scores_read=False, API_requests_by_this_check=0)
    (run/'launch_validation.json').write_text(json.dumps(result, indent=2))
    state_path = PROJECT/'reports/PIPELINE_STATE.json'; state = read(state_path)
    mpe = state['mpe']
    mpe['mappo40_formal'] = {k: mpe[k] for k in ('status', 'run', 'analysis', 'actual_train_env_steps', 'episodes')}
    relative = str(run.relative_to(PROJECT))
    mpe.update(status='soft_running', run=relative, analysis=None, actual_train_env_steps=None,
        episodes=None, container=container, exec_session=session, teacher_formal_started=True,
        active_protocol=c['protocol_document'], protocol=c['protocol_document'], auto_restart=False,
        teacher_cutoff_step=100000, API_concurrency=1000, guard_max_in_flight=1000,
        cache_first=True, last_annotated_update_step=99000)
    state.update(formal_experiments='mpe_soft_running', formal_run=relative,
        formal_container=container, formal_exec_session=session,
        monitoring='one_launch_metadata_check_then_natural_completion',
        current_tests='reports/mpe_soft1000_20261020/VALIDATION.md',
        updated_at=datetime.now(timezone.utc).isoformat())
    state_path.write_text(json.dumps(state, indent=2))
    return result


def completion(run):
    frozen = read(run/'config.json'); c, _ = verify_config(frozen)
    status = read(run/'status.json'); analysis = Path(str(run)+'_analysis')
    assert status['status'] == 'complete' and status['train_env_steps'] == 1000000 and status['episodes'] == 40000
    assert read(analysis/'validation.json')['status'] == 'passed'
    for ext in ('png', 'pdf', 'svg'):
        assert (analysis/f'learning_curves.{ext}').stat().st_size > 0
    target_batches = Counter(); valid = masked = cache_hits = 0
    for line in (run/'teacher_targets.jsonl').open():
        target = json.loads(line); step = target['actual_train_env_steps']
        assert step < c['kd_anneal_t2'] and step % 1000 == 0
        target_batches[step] += 1
        valid += target['teacher_valid']; masked += 1-target['teacher_valid']
        cache_hits += int(target['cache_hit'])
    assert target_batches == Counter({step: 1000 for step in range(1000, 100000, 1000)})
    http = reasoning_responses = 0; usage = Counter()
    audit = run/'teacher_calls.jsonl'
    if audit.exists():
        for line in audit.open():
            call = json.loads(line); http += 1
            assert call['endpoint'] == c['teacher']['endpoint']
            assert call['thinking'] == dict(type='disabled')
            assert call['chat_template_kwargs'] == dict(thinking=False, enable_thinking=False)
            assert call['top_p_requested'] == .95 and call['max_tokens'] is None
            reasoning_responses += call.get('off_response_evidence') == 'returned_reasoning'
            reported = call.get('response', {}).get('usage', {})
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                usage[key] += reported.get(key) or 0
            usage['reasoning_tokens'] += (reported.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
    assert http == status['teacher_calls'] and status['teacher_probability_cache_enabled']
    result = dict(status='passed', teacher_target_states=valid+masked, valid_states=valid,
        masked_states=masked, cache_hit_targets=cache_hits, API_requests=http,
        responses_with_reasoning=reasoning_responses, usage_totals=dict(usage),
        teacher_cutoff_step=100000, last_annotated_update_step=99000,
        no_teacher_targets_at_or_after_cutoff=True, teacher_batches=len(target_batches),
        API_requests_during_validation=0,
        note='Usage totals sum reported fields, not a verified invoice; evaluation is student-only.')
    (analysis/'teacher_validation.json').write_text(json.dumps(result, indent=2))
    state_path = PROJECT/'reports/PIPELINE_STATE.json'; state = read(state_path)
    relative = str(run.relative_to(PROJECT))
    if state['mpe'].get('run') == relative:
        state['mpe'].update(status='soft_complete_validated', container=None, exec_session=None,
            actual_train_env_steps=1000000, episodes=40000, analysis=str(analysis.relative_to(PROJECT)),
            teacher_validation=str((analysis/'teacher_validation.json').relative_to(PROJECT)))
        state.update(formal_experiments='mpe_soft_complete_validated', formal_container=None,
            formal_exec_session=None, updated_at=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(state, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['launch', 'completion']); p.add_argument('run', type=Path)
    p.add_argument('--container'); p.add_argument('--session', type=int)
    a = p.parse_args(); run = a.run.resolve()
    if not run.is_relative_to(PROJECT.resolve()): raise ValueError('Project-owned output required')
    if a.action == 'launch':
        if not a.container or a.session is None: p.error('launch requires container/session')
        result = launch(run, a.container, a.session)
    else: result = completion(run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
