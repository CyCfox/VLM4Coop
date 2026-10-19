"""Audit and export one completed MAPPO baseline, without implying comparisons."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from vlm4coop.analysis import load_run, read_jsonl
from vlm4coop.config import PROJECT
from vlm4coop.metrics import summarize
from vlm4coop.scenario_sampling import audit_seed_pool


def close(a, b):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    return b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if not run.is_relative_to(PROJECT.resolve()) or not output.is_relative_to(PROJECT.resolve()):
        raise ValueError('Input and output must stay inside VLM4Coop')
    data = load_run(run, 'mappo', 3)  # Reject running, smoke, or incomplete runs.
    config, status = data['config'], data['status']
    protocol = config['experiment_config']
    epymarl = config['epymarl_args']
    assert protocol['environment']['field_size'] == [8, 8]
    assert config['seed'] == epymarl['seed'] == 0 and protocol['train_seeds'] == [0]
    assert protocol['fixed_eval_seeds'] == {'start': 1000000, 'count': 100}
    assert protocol['eval_episodes_every_checkpoint'] == 100
    assert epymarl['epochs'] == 4 and epymarl['batch_size_run'] == 10
    assert not epymarl['test_greedy'] and not epymarl['use_cuda']
    assert status['teacher_calls'] == 0
    pool_validation = audit_seed_pool(run)
    training = read_jsonl(run / 'train_episodes.jsonl')
    assert len(training) == status['episodes']
    assert sum(row['length'] for row in training) == status['train_env_steps']
    batches = Counter(row['actual_train_env_steps'] for row in training)
    assert set(batches.values()) == {10}, batches
    cumulative = 0
    for actual in sorted(batches):
        cumulative += sum(row['length'] for row in training if row['actual_train_env_steps'] == actual)
        assert cumulative == actual
    episodes = read_jsonl(run / 'eval_episodes.jsonl')
    for saved in data['evaluations']:
        assert saved['train_seed'] == 0
        selected = [row for row in episodes if row['scheduled_eval_step'] == saved['scheduled_eval_step']]
        start = protocol['fixed_eval_seeds']['start']
        assert sorted(row['eval_seed'] for row in selected) == list(range(start, start + 100))
        assert all(row['actual_train_env_steps'] == saved['actual_train_env_steps'] for row in selected)
        assert all(row['train_seed'] == 0 for row in selected)
        assert all(row['success'] == (row['remaining_food'] == 0) for row in selected)
        assert all(0 < row['length'] <= protocol['environment']['max_episode_steps'] for row in selected)
        for key, value in summarize(selected).items():
            assert close(value, saved[key]), (saved['scheduled_eval_step'], key, value, saved[key])
    scalars = read_jsonl(run / 'scalars.jsonl')
    assert all(math.isfinite(row['value']) for row in scalars), 'Nonfinite training scalar'
    final_step = status['train_env_steps']
    assert data['final']['actual_train_env_steps'] == final_step
    assert (run / 'checkpoints' / str(final_step) / 'agent.th').stat().st_size > 0
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'learning_curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data['evaluations'][0]))
        writer.writeheader()
        writer.writerows({key: 'NaN' if value is None else value for key, value in row.items()}
                         for row in data['evaluations'])
    (output / 'final_metrics.json').write_text(json.dumps(data['final'], indent=2, allow_nan=False))
    files = ['config.json', 'status.json', 'provenance.json', 'source_snapshot.tar.gz',
             'train_episodes.jsonl', 'eval_episodes.jsonl', 'evaluations.jsonl', 'scalars.jsonl']
    if pool_validation is not None:
        files += ['training_seed_pool.json', 'train_batches.jsonl']
    evidence = dict(status='passed', method='mappo', formal=True, source=str(run),
                    actual_train_env_steps=final_step, training_episodes=len(training),
                    evaluations=len(data['evaluations']), raw_eval_episodes=len(episodes),
                    teacher_calls=0, all_raw_statistics_recomputed=True, training_seed_pool=pool_validation,
                    source_hashes={name: sha256((run / name).read_bytes()).hexdigest() for name in files},
                    analysis_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'validation.json').write_text(json.dumps(evidence, indent=2))
    plot(data['evaluations'], output, pool_validation)
    final = data['final']
    lines = ['# MAPPO 8×8 baseline', '',
             'Native EPyMARL MAPPO; 3 agents/3 food, seed 0, stochastic evaluation. ',
             f'Completed {final_step:,} joint environment steps; {len(episodes):,} raw evaluation episodes.', '',
             '| Final checkpoint metric | Mean | Sample SD | Episodes |',
             '|---|---:|---:|---:|']
    for metric, label, n in [('success_rate', 'Success rate', 100), ('return', 'Team episode return', 100),
                             ('success_episodes_length', 'Completion steps (successes only)', final['n_success_episodes']),
                             ('episodes_length', 'Episode steps (all)', 100)]:
        values = [final[metric + '_mean'], final[metric + '_std']]
        formatted = ['undefined' if value is None else f'{value:.6f}' for value in values]
        lines.append(f'| {label} | {formatted[0]} | {formatted[1]} | {n} |')
    lines.extend(['', 'SD describes episode variation for a single trained policy, not variation across training seeds.',
                  'The final checkpoint is reported; selecting the best checkpoint on these same scenarios would not be an unbiased final test.',
                  'This baseline alone cannot establish an advantage for VLM4Coop. No teacher-method comparison is available yet.'])
    if pool_validation is not None:
        lines.extend(['', f"Training used {pool_validation['pool_size']} fixed scenario seeds, sampled uniformly in batches of 10 without replacement within each batch.",
                      f"Unique training seeds observed: {pool_validation['unique_training_seeds_seen']}. Evaluation seeds are disjoint from the training pool."])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(validation=evidence, final=final)))


def plot(rows, output, pool_validation=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'], 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 1.2, 'legend.frameon': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    x = np.asarray([row['actual_train_env_steps'] for row in rows])
    for ax, metric, ylabel in zip(axes, ('success_rate', 'return'), ('Success rate', 'Team episode return')):
        mean = np.asarray([row[metric + '_mean'] for row in rows])
        sd = np.asarray([row[metric + '_std'] for row in rows])
        ax.plot(x, mean, color='#4D4D4D', lw=1.5, label='MAPPO (seed 0)')
        ax.fill_between(x, mean - sd, mean + sd, color='#767676', alpha=.18, linewidth=0,
                        label='Episode mean ± SD')
        ax.set(xlabel='Environment steps', ylabel=ylabel, xlim=(0, max(x)))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.legend(fontsize=7.5, loc='upper left')
    fig.tight_layout(pad=1)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(output / f'mappo_learning_curves.{extension}', dpi=300)
    plt.close(fig)
    caption = (
        'Native MAPPO on 8×8 LBF with 3 agents and 3 food, one training seed (0). '
        'Each point is the mean of 100 stochastic evaluation episodes on fixed seeds 1000000–1000099; '
        'shading is ±1 sample SD across episodes (ddof=1), not a confidence interval or training-seed variation. '
        'Both scheduled and actual steps are provided in learning_curves.csv; the x axis uses actual joint environment steps. '
        'No smoothing, interpolation, or best-checkpoint selection. Mean±SD bands may extend beyond the metric bounds.\n')
    if pool_validation is not None:
        caption += (f"Initial training scenarios are restricted to {pool_validation['pool_size']} fixed seeds, "
                    'with 10 distinct scenarios sampled uniformly per rollout batch; scenarios may repeat across batches. '
                    'Evaluation seeds are disjoint from the training pool.\n')
    (output / 'FIGURE_CAPTION.txt').write_text(caption)


if __name__ == '__main__':
    main()
