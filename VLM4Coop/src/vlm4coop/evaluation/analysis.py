"""Export only completed, genuine formal runs. Never synthesize missing curves."""
import argparse
import csv
import json
from pathlib import Path
from vlm4coop.training.configuration import PROJECT


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_run(path, expected_method, agents):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    status = json.loads((path / 'status.json').read_text())
    if status['status'] != 'complete' or config['profile'] != 'formal':
        raise ValueError(f'{path}: incomplete or non-formal run')
    if config['method'] != expected_method:
        raise ValueError(f'{path}: wrong method')
    if expected_method != 'pure_vlm' and config['agents'] != agents:
        raise ValueError(f'{path}: wrong scenario')
    evaluations = read_jsonl(path / 'evaluations.jsonl')
    episodes = read_jsonl(path / 'eval_episodes.jsonl')
    if expected_method == 'pure_vlm':
        if len(evaluations) != 1 or len(episodes) != 100:
            raise ValueError('Pure VLM needs 100 real episodes')
    else:
        budget = config['experiment_config']['formal_train_steps']
        interval = config['experiment_config']['eval_interval_steps']
        if status['train_env_steps'] < budget:
            raise ValueError('Formal training budget not reached')
        if [e['scheduled_eval_step'] for e in evaluations] != list(range(interval, budget + 1, interval)):
            raise ValueError('Missing or duplicate scheduled evaluation')
        if len(episodes) != len(evaluations) * 100:
            raise ValueError('Evaluation episode count mismatch')
    return {'path': str(path), 'config': config, 'status': status,
            'evaluations': evaluations, 'final': evaluations[-1]}


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(mapping, output):
    spec = [('mappo_3', 'mappo', 3), ('soft_3', 'soft', 3), ('hard_3', 'hard', 3),
            ('soft_4', 'soft', 4), ('soft_5', 'soft', 5), ('pure_vlm_3', 'pure_vlm', 3)]
    runs = {key: load_run(mapping[key], method, agents) for key, method, agents in spec}
    # Protect against accidentally mixing protocol generations/library stacks.
    reference = runs['mappo_3']['config']['experiment_config']
    def student_protocol(config):
        # Baseline never calls a teacher. Deployment authorization is metadata.
        return {key: value for key, value in config.items()
                if key not in {'teacher', 'protocol_status', 'formal_authorized_methods'}}
    teacher_reference = runs['soft_3']['config']['experiment_config']['teacher']
    for run in runs.values():
        current = run['config']['experiment_config']
        if student_protocol(current) != student_protocol(reference):
            raise ValueError('Formal runs use different experiment protocols')
        if run['config']['method'] != 'mappo' and current['teacher'] != teacher_reference:
            raise ValueError('Teacher methods use different frozen teacher protocols')
    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT.resolve()):
        raise ValueError('Analysis output must stay in VLM4Coop')
    output.mkdir(parents=True, exist_ok=False)
    curves = [dict(run=key, **row) for key, run in runs.items() for row in run['evaluations']]
    _csv(output / 'learning_curves.csv', curves)
    final = [dict(run=key, **run['final']) for key, run in runs.items()]
    _csv(output / 'final_metrics.csv', final)
    (output / 'evidence.json').write_text(json.dumps({key: {'source': run['path'], 'status': run['status']}
                                                       for key, run in runs.items()}, indent=2))
    lines = [r'\begin{tabular}{lrrr}', r'\hline',
             r'Method & Success rate & Team return & Steps (success) \\', r'\hline']
    for key, label in [('mappo_3', 'MAPPO'), ('pure_vlm_3', 'Pure VLM'), ('soft_3', 'VLM4Coop')]:
        summary = runs[key]['final']
        values = []
        for metric in ('success_rate', 'return', 'success_episodes_length'):
            mean, std = summary[metric + '_mean'], summary[metric + '_std']
            values.append('--' if mean is None else f'${mean:.3f} \\pm {std:.3f}$' if std is not None else f'${mean:.3f}$')
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    lines.extend([r'\hline', r'\end{tabular}'])
    (output / 'main_results.tex').write_text('\n'.join(lines) + '\n')
    plot(runs, output)
    return output


def plot(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    # scientific-figure-making house style, compact IEEE two-column sizing.
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'], 'font.size': 9,
                         'axes.spines.right': False, 'axes.spines.top': False,
                         'axes.linewidth': 1.2, 'legend.frameon': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = {'mappo_3': '#767676', 'soft_3': '#0F4D92', 'hard_3': '#B64342',
              'soft_4': '#42949E', 'soft_5': '#9A4D8E'}
    panels = [('main', [('mappo_3', 'MAPPO'), ('soft_3', 'VLM4Coop')]),
              ('ablation', [('soft_3', 'Soft'), ('hard_3', 'Hard')]),
              ('scale', [('soft_3', '3 agents/3 food'), ('soft_4', '4 agents/4 food'),
                         ('soft_5', '5 agents/5 food')])]
    for name, selections in panels:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
        for ax, metric, ylabel in zip(axes, ('success_rate', 'return'), ('Success rate', 'Average team episode return')):
            for key, label in selections:
                data = runs[key]['evaluations']
                x = np.asarray([e['actual_train_env_steps'] for e in data])
                y = np.asarray([e[metric + '_mean'] for e in data])
                std = np.asarray([e[metric + '_std'] for e in data])
                ax.plot(x, y, color=colors[key], lw=1.5, label=label)
                ax.fill_between(x, y - std, y + std, color=colors[key], alpha=.15, linewidth=0)
            ax.set_xlabel('Environment steps')
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            ax.legend(fontsize=8)
        fig.tight_layout(pad=1)
        for extension in ('pdf', 'svg', 'png'):
            fig.savefig(output / f'{name}.{extension}', dpi=300)
        plt.close(fig)
    (output / 'FIGURE_CAPTIONS.txt').write_text(
        'All curves: one training seed (0); means and shaded ±1 sample standard deviation over 100 fixed evaluation episodes. '
        'Shading is episode variability, not training-seed variability or a confidence interval. '
        'No smoothing or interpolation; x is actual training environment steps. '
        'Each scale is independently trained. Pure VLM has no training curve.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True, help='JSON mapping of six required run keys to directories')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(analyze(json.loads(Path(args.runs).read_text()), args.output))
