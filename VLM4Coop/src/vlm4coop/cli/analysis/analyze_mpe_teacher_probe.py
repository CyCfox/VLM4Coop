"""Offline replay/audit of one completed 3x25 MPE teacher probe; no model calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from vlm4coop.cli.probes.mpe_teacher_probe import world_state,dump
from vlm4coop.envs.mpe.adapter import MPEGymBridge
from vlm4coop.evaluation.analysis import read_jsonl
from vlm4coop.teachers.api import completion_text,off_response_evidence
from vlm4coop.teachers.response_protocol import parse_api_probabilities
from vlm4coop.utils.paths import PROJECT


def analyze(run):
    run=Path(run).resolve()
    if not run.is_relative_to((PROJECT/'results').resolve()): raise ValueError('Project result required')
    config=json.loads((run/'config.json').read_text()); c=config['experiment_config']
    status=json.loads((run/'status.json').read_text())
    assert status['status']=='complete' and status['completed_joint_decisions']==75
    assert config['profile']=='probe' and not config['formal'] and status['cache_enabled'] is False
    steps=read_jsonl(run/'pure_vlm_steps.jsonl'); trace=read_jsonl(run/'trajectory.jsonl')
    calls=read_jsonl(run/'teacher_calls.jsonl'); timings=read_jsonl(run/'batch_timings.jsonl')
    episodes=read_jsonl(run/'eval_episodes.jsonl')
    assert len(steps)==len(trace)==75 and len(timings)==25 and len(episodes)==3
    assert len(calls)==status['teacher_calls'] and status['peak_active_http_requests']<=3
    assert all(t['active_environments']==3 for t in timings)
    by_key={(s['worker'],s['t']):s for s in steps}
    assert len(by_key)==75 and len(list((run/'frames').glob('*.png')))==75
    for event in calls:
        assert event['image_count']==1 and event['system']==config['system_prompt'] and event['prompt']==config['user_prompt']
        assert event['model_requested']==c['teacher']['model'] and event['endpoint']==c['teacher']['endpoint']
        assert event['top_p_requested']==.95 and event['thinking']==dict(type='disabled') and event['max_tokens'] is None
        if 'chat_template_kwargs' in c['teacher']:
            assert event.get('chat_template_kwargs')==c['teacher']['chat_template_kwargs']
            assert event['reasoning_effort']=='provider_default'
    square = 'square1440' in json.loads((run/'cache_identity.json').read_text()).get('renderer','')
    if square:
        for folder in ('frames','initial','final'):
            for path in (run/folder).glob('*.png'):
                with Image.open(path) as image: assert image.size==(1440,1440)
    fallbacks=0
    for s in steps:
        p=np.asarray(s['probabilities'],dtype=np.float32)
        assert p.shape==(3,5) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert np.allclose(p.sum(-1),1) and p.argmax(-1).tolist()==s['actions'] and not s['cache_hit']
        if s['source']=='pure_none_fallback':
            fallbacks+=1
            assert s['format_attempts']==3 and s['actions']==[0,0,0]
        else:
            parsed,_=parse_api_probabilities(s['raw'],3,purpose='pure',n_actions=5)
            np.testing.assert_array_equal(parsed,p)
            original=[e for e in calls if '_'+s['decision_id']+'_format' in e['label'] and e.get('response')]
            assert original and completion_text(original[-1]['response'])[0]==s['raw']
    assert fallbacks==status['none_fallback_decisions']
    per_seed=[]
    for worker,seed in enumerate(config['environment_seeds']):
        rows=[r for r in trace if r['worker']==worker]
        assert [r['t'] for r in rows]==list(range(25)) and all(r['env_seed']==seed for r in rows)
        env=MPEGymBridge(**c['environment']); total=0.
        try:
            env.reset(seed=seed)
            for row in rows:
                assert world_state(env)==row['before']
                if square:
                    with Image.open(run/'frames'/row['frame_file']) as image:
                        np.testing.assert_array_equal(np.asarray(image.convert('RGB')),env.render())
                assert by_key[(worker,row['t'])]['actions']==row['actions']
                _,reward,terminated,truncated,info=env.step(row['actions'])
                assert world_state(env)==row['after'] and not terminated and bool(truncated)==(row['t']==24)
                assert abs(float(sum(reward))-row['reward'])<1e-9
                total+=sum(reward)
            episode=next(e for e in episodes if e['env_seed']==seed)
            assert abs(total-episode['team_return'])<1e-9 and episode['length']==25
            final=env.state_metrics()
            assert final['covered_landmarks']==episode['final_covered_landmarks']
            assert bool(final['success'])==episode['success']
            per_seed.append(dict(seed=seed,team_return=total,initial_mean_min_distance=rows[0]['before']['metrics']['mean_min_distance'],
                final_mean_min_distance=final['mean_min_distance'],final_covered_landmarks=final['covered_landmarks'],
                final_success=bool(final['success']),first_all_covered_step=env.first_all_covered_step,
                collision_pairs_total=env.collision_pairs_total))
        finally: env.close()
    usage=[e.get('response',{}).get('usage',{}) for e in calls]
    def token_sum(key):
        values=[u[key] for u in usage if isinstance(u.get(key),(int,float))]
        return sum(values) if values else None
    reasoning=[u.get('completion_tokens_details',{}).get('reasoning_tokens') for u in usage]
    reasoning=[x for x in reasoning if isinstance(x,(int,float))]
    evidence=Counter(off_response_evidence(e['response']) for e in calls if e.get('response',{}).get('choices'))
    audit=dict(status='passed',API_requests_during_analysis=0,native_trajectory_steps_replayed=75,
        original_HTTP_attempts=len(calls),HTTP_status_counts=dict(Counter(str(e.get('http_status')) for e in calls)),
        accepted_response_probabilities_reparsed=True,raw_returns_and_coverage_replayed=True,
        cache_hits=0,cache_writes=0,none_fallback_decisions=fallbacks,
        format_attempts_histogram=dict(Counter(str(s['format_attempts']) for s in steps)),
        models_returned=sorted({str(e.get('response',{}).get('model')) for e in calls}),
        off_response_evidence=dict(evidence),
        fixed_square_images_verified=square,RGB_native_replay_verified=square,
        usage=dict(prompt_tokens=token_sum('prompt_tokens'),completion_tokens=token_sum('completion_tokens'),
                   total_tokens=token_sum('total_tokens'),reasoning_tokens=sum(reasoning) if reasoning else None),
        HTTP_latency_mean_seconds=float(np.mean([e['latency_seconds'] for e in calls])),
        HTTP_latency_p95_seconds=float(np.percentile([e['latency_seconds'] for e in calls],95)),
        per_seed=per_seed)
    output=run/'analysis'; output.mkdir(exist_ok=False)
    dump(output/'audit.json',audit)
    (output/'analysis_source.py').write_text(Path(__file__).read_text())
    lines=['# MPE teacher:3parallel environments ×25steps','',
        'This is a three-episode capability probe, not the formal100scenario evaluation.',
        'Native trajectories/returns/coverage were replayed offline; no analysis API calls.','',
        '| Seed | Native team return | Final covered goals | Mean nearest-agent distance (initial → final) | Final all3coverage |',
        '|---|---:|---:|---:|---|']
    for r in per_seed:
        lines.append(f"| {r['seed']} | {r['team_return']:.4f} | {r['final_covered_landmarks']}/3 | {r['initial_mean_min_distance']:.4f} → {r['final_mean_min_distance']:.4f} | {r['final_success']} |")
    lines+=['',f"Wall time {status['wall_seconds']:.3f}s; HTTP attempts{len(calls)}; peak concurrent requests{status['peak_active_http_requests']}; NONEfallbacks{fallbacks}.",
        'Requested OFF is a preference; raw response usage determines whether hidden reasoning actually occurred.',
        f"Observed OFF evidence: {dict(evidence)}. Completion tokens: {audit['usage']['completion_tokens']}; reasoning tokens: {audit['usage']['reasoning_tokens']}.",
        'Missing landmark labels do not establish geometric coverage; metrics above use true distance<0.1, independent of prompt assertions.',
        'Fixed25steps is not completion time. This small probe does not establish superiority or cross-seed training stability.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,ensure_ascii=False))
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    analyze(parser.parse_args().run)
