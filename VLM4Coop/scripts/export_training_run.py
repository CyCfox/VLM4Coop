"""Offline exports for one completed formal training run; no teacher requests."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from vlm4coop.analysis import load_run, read_jsonl
from vlm4coop.config import PROJECT
from vlm4coop.metrics import summarize
from vlm4coop.scenario_sampling import audit_seed_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    run, output = options.run.resolve(), options.output.resolve()
    if not all(path.is_relative_to(PROJECT.resolve()) for path in (run, output)):
        raise ValueError('All paths must remain under VLM4Coop')
    frozen = json.loads((run / 'config.json').read_text())
    if frozen['method'] not in ('mappo', 'soft', 'hard'):
        raise ValueError('Expected a training method')
    data = load_run(run, frozen['method'], frozen['agents'])
    pool = audit_seed_pool(run)
    training = read_jsonl(run / 'train_episodes.jsonl')
    assert len(training) == data['status']['episodes']
    assert sum(row['length'] for row in training) == data['status']['train_env_steps']
    records = read_jsonl(run / 'eval_episodes.jsonl')
    for result in data['evaluations']:
        selected = [row for row in records if row['scheduled_eval_step'] == result['scheduled_eval_step']]
        for name, value in summarize(selected).items():
            saved = result[name]
            assert (not math.isfinite(value) and saved is None) or (saved is not None and math.isclose(value, saved, abs_tol=1e-9, rel_tol=1e-9))
    assert all(math.isfinite(row['value']) for row in read_jsonl(run / 'scalars.jsonl'))
    valid = masked = 0
    targets = run / 'teacher_targets.jsonl'
    if targets.exists():
        for line in targets.open():
            row = json.loads(line)
            if row['teacher_valid']:
                p = np.asarray(row['probabilities'])
                assert p.shape == (frozen['agents'], 6) and np.isfinite(p).all()
                assert (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(-1), 1)
                valid += 1
            else:
                assert row['source'] == 'kd_masked' and row['probabilities'] is None and row['format_attempts'] == 3
                masked += 1
    final_step = data['status']['train_env_steps']
    assert (run / 'checkpoints' / str(final_step) / 'agent.th').stat().st_size > 0
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'learning_curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data['evaluations'][0]))
        writer.writeheader()
        writer.writerows(data['evaluations'])
    (output / 'final_metrics.json').write_text(json.dumps(data['final'], indent=2, allow_nan=False))
    evidence = dict(status='passed', formal=True, method=frozen['method'], agents=frozen['agents'],
        run=str(run), train_env_steps=final_step, training_seed_pool=pool,
        evaluations=len(data['evaluations']), valid_teacher_states=valid,
        masked_teacher_states=masked, api_requests_during_analysis=0,
        note='Episode sample SD (ddof=1); successful episodes only for completion time. No method superiority is inferred.')
    (output / 'validation.json').write_text(json.dumps(evidence, indent=2, allow_nan=False))
    (output / 'REPORT.md').write_text(
        f"# Completed {frozen['method']} training export\n\n"
        f"Source: {run}\n\nActual joint training steps: {final_step}. "
        f"Evaluations: {len(data['evaluations'])}. "
        f"Valid/masked teacher-supervised states: {valid}/{masked}.\n\n"
        'Raw evaluation statistics were recomputed offline. learning_curves.csv preserves scheduled and actual steps without smoothing. '
        'Means and sample SD refer to evaluation episodes, not training seeds. '
        'Final metrics are in final_metrics.json. No teacher calls were made during analysis; no comparison claim is made.\n')
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
