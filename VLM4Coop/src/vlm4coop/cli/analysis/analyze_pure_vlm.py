"""Recompute an uncached Pure VLM evaluation from raw episode and action logs."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import math
import numpy as np
from vlm4coop.evaluation.metrics import summarize
from vlm4coop.teachers.response_protocol import parse_api_probabilities, none_probabilities
from vlm4coop.teachers.api import off_response_evidence, completion_text
from vlm4coop.utils.json_logging import _json_safe


def rows(path):
    return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    run = a.run
    frozen = json.loads((run/'config.json').read_text())
    config = frozen['experiment_config']
    assert frozen['profile']=='formal' and 'test_double' not in config
    status = json.loads((run/'status.json').read_text())
    if status['status']!='complete':
        partial = json.loads((run/'partial_episodes.json').read_text()) if (run/'partial_episodes.json').exists() else []
        report = dict(status='incomplete', original_status=status,
            completed_episodes=sum(row is not None for row in partial),
            recorded_decisions=len(rows(run/'pure_vlm_steps.jsonl')),
            note='The formal 100-episode evaluation did not complete; no full-evaluation score is reported.')
        (run/'validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return 1
    assert config['teacher']['pure_vlm_use_cache'] is False
    is_mpe=config.get('environment_adapter','lbf')=='mpe_simple_spread'
    horizon=config['environment']['max_episode_steps']
    actions=frozen['epymarl_args']['n_actions']
    assert config['eval_parallel_envs']==config['pure_vlm_parallel_envs']
    assert config['teacher']['concurrency']==config['pure_vlm_concurrency']
    assert frozen['epymarl_args']['batch_size_run']==config['eval_parallel_envs']
    episodes = rows(run/'eval_episodes.jsonl')
    expected = list(range(config['fixed_eval_seeds']['start'],config['fixed_eval_seeds']['start']+100))
    assert len(episodes)==100 and [row['env_seed'] for row in episodes]==expected
    assert all(row['env_seed']==row['eval_seed'] for row in episodes)
    if is_mpe:
        assert all(row['length']==25 and row['episode_limit'] and row['success']==(row['final_covered_landmarks']==3) for row in episodes)
    else:
        assert all(row['initial_food_count']==3 for row in episodes)
        assert all(0<row['length']<=horizon and row['success']==(row['remaining_food']==0) for row in episodes)
    steps = rows(run/'pure_vlm_steps.jsonl')
    assert len(steps)==sum(row['length'] for row in episodes)==status['decisions']
    for episode in episodes:
        selected=[row for row in steps if row['env_seed']==episode['env_seed']]
        assert [row['t'] for row in selected]==list(range(episode['length']))
        assert all(row['worker']==episode['worker'] for row in selected)
    assert not any(row['cache_hit'] for row in steps)
    for row in steps:
        if row.get('source','teacher')=='pure_none_fallback':
            probs,repaired=none_probabilities(3,actions),False
            assert row['format_attempts']==3
        else:
            assert row.get('source','teacher')=='teacher'
            probs,repaired=parse_api_probabilities(row['raw'],3,'pure',actions)
        np.testing.assert_array_equal(probs,np.asarray(row['probabilities'],dtype=np.float32))
        assert repaired==row['repaired'] and probs.argmax(-1).tolist()==row['actions']
        frame=row.get('frame_file',f"worker{row['worker']:03d}_t{row['t']:03d}.png")
        assert (run/'frames'/frame).is_file()
    assert len(list((run/'frames').glob('*.png')))==len(steps)
    calls=rows(run/'teacher_calls.jsonl')
    labels=rows(run/'teacher_labels.jsonl')
    teacher_steps=[row for row in steps if row.get('source','teacher')=='teacher']
    fallback_steps=[row for row in steps if row.get('source')=='pure_none_fallback']
    assert len(calls)==status['teacher_calls'] and len(labels)==len(teacher_steps)
    assert len(fallback_steps)==status['none_fallback_decisions']
    assert all(label['key'] is None for label in labels)
    successful={row['response']['id']:row for row in calls if row.get('valid_transport_completion')}
    accepted=[]
    for label in labels:
        event=successful[label['response_id']]
        message=event['response']['choices'][0]['message']
        evidence=off_response_evidence(event['response'])
        assert label['thinking_evidence']==evidence
        assert label['reasoning_tokens']==event['response'].get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens')
        if label.get('control_mode') not in ('request_off_accept_returned_reasoning','verified_off_with_rate_guard_v1'):
            assert evidence!='returned_reasoning'
        assert event['thinking']=={'type':'disabled'} and event['top_p_requested']==.95
        if 'chat_template_kwargs' in config['teacher']:
            assert event.get('chat_template_kwargs')==config['teacher']['chat_template_kwargs']
            assert event['reasoning_effort']=='provider_default'
        raw,answer_source=completion_text(event['response'])
        if 'answer_source' in label: assert label['answer_source']==answer_source
        accepted.append(raw)
    assert Counter(accepted)==Counter(row['raw'] for row in teacher_steps)
    failures=rows(run/'format_failures.jsonl')
    failed_counts=Counter(row['decision_id'] for row in failures)
    assert len(failures)==status['format_failures']
    assert len({row['decision_id'] for row in steps})==len(steps)
    for row in steps:
        expected_failures=3 if row['source']=='pure_none_fallback' else row['format_attempts']-1
        assert failed_counts[row['decision_id']]==expected_failures
    assert status['cache_hits']==status['cache_misses']==status['cache_writes']==0
    recomputed=_json_safe(summarize(episodes))
    saved=rows(run/'evaluations.jsonl')
    assert len(saved)==1
    for name,value in recomputed.items():
        assert saved[0][name]==value
    report=dict(status='passed',statistics=recomputed,
        fixed_environment_seeds_verified=True,all_frame_step_episode_counts_match=True,
        actual_environment_steps=len(steps),accepted_labels=len(labels),
        http_attempts=len(calls),peak_active_http_requests=status['peak_active_http_requests'],
        probability_cache_reads=0,probability_cache_writes=0,
        reported_zero_reasoning_labels=sum(label['thinking_evidence']=='reported_zero' for label in labels),
        reasoning_usage_unreported_labels=sum(label['thinking_evidence']=='not_reported' for label in labels),
        returned_reasoning_labels=sum(label['thinking_evidence']=='returned_reasoning' for label in labels),
        none_fallback_decisions=len(fallback_steps),format_failures=len(failures),
        format_retries=status['format_retries'],
        http_status_counts=dict(Counter(str(row.get('http_status','transport_error')) for row in calls)),
        wall_seconds=status['wall_seconds'],api_requests_during_analysis=0)
    if is_mpe:
        from vlm4coop.evaluation.mpe_replay import replay_pure_mpe
        report['native_RGB_replay']=replay_pure_mpe(run,config,episodes,steps)
        report['requested_controls_verified']=True
        for event in calls:
            assert event['endpoint']==config['teacher']['endpoint']
            assert event['model_requested']==config['teacher']['model']
            assert event['max_tokens'] is None
    def token_sum(name):
        values=[r.get('response',{}).get('usage',{}).get(name) for r in calls]
        return sum(v for v in values if isinstance(v,(int,float)))
    report['usage']=dict(prompt_tokens=token_sum('prompt_tokens'),completion_tokens=token_sum('completion_tokens'),
                         total_tokens=token_sum('total_tokens'),
        reasoning_tokens=sum(r.get('response',{}).get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0) or 0 for r in calls))
    (run/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (run/'episodes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episodes[0]))
        writer.writeheader();writer.writerows(episodes)
    (run/'statistics.json').write_text(json.dumps(recomputed,indent=2,allow_nan=False))
    with (run/'statistics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(recomputed))
        writer.writeheader();writer.writerow(recomputed)
    def fmt(value): return 'N/A' if value is None else f'{value:.4f}'
    scenario=('MPE simple_spread, 3 agents/3 landmarks, 25-step fixed horizon' if is_mpe
              else '8×8, 3 agents/3 foods, 100-step limit')
    (run/'REPORT.md').write_text(
        '# Pure VLM formal evaluation: uncached, 100 fixed environments\n\n'
        f"Completed {len(episodes)} episodes on seeds {expected[0]}..{expected[-1]}, {scenario}.\n\n"
        f"- Success: {recomputed['n_success_episodes']}/100; rate {fmt(recomputed['success_rate_mean'])} ± {fmt(recomputed['success_rate_std'])}.\n"
        f"- Team return: {fmt(recomputed['return_mean'])} ± {fmt(recomputed['return_std'])}.\n"
        f"- Successful-episode completion steps: {fmt(recomputed['success_episodes_length_mean'])} ± {fmt(recomputed['success_episodes_length_std'])}; failures excluded.\n"
        f"- Joint environment steps: {len(steps)}; HTTP attempts including retries/controls: {len(calls)}; peak active client requests: {status['peak_active_http_requests']}.\n"
        f"- Wall time: {status['wall_seconds']:.3f} s. Probability-cache reads/writes: 0/0.\n\n"
        f"- Format retries: {status['format_retries']}; all-agent NONE defaults after three failures: {len(fallback_steps)}. These defaults are controller actions, not model probability labels.\n\n"
        f"- Thinking requested OFF for all accepted labels: {report['reported_zero_reasoning_labels']} report zero reasoning tokens; {report['reasoning_usage_unreported_labels']} omit this usage field; {report['returned_reasoning_labels']} contain returned reasoning. Missing telemetry is not proof of zero internal reasoning.\n\n"
        'Means and sample standard deviations (ddof=1) are across episodes, not model training seeds. '
        'Missing success-time statistics are N/A (JSON null). '
        + ('MPE velocity-vector prompts and RGB; final coverage success is supplementary, not early termination; fixed-horizon length is not completion time. ' if is_mpe else
         'Frozen v6 prompts, one global RGB with stable F0 food IDs, no supplied current coordinates. ')+
        'Thinking requested disabled, top_p=0.95 requested, no client max_tokens; per-agent argmax, with all-agent NONE after three format failures and no external planner. '
        'Every effective state makes a fresh inference; repeated images are not deduplicated. '
        'All PNG/step/episode/response/usage records and recomputed statistics were checked offline; no new API requests. '
        'Source/config/dependency snapshots are saved in this directory.\n')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
