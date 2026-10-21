"""Plot genuine checkpoint summaries; no smoothing or model inference."""
import json
from pathlib import Path
import numpy as np


def plot_training_curves(rows, frozen, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import __version__ as pillow_version

    output = Path(output)
    names = [f'learning_curves.{ext}' for ext in ('png', 'pdf', 'svg')]
    if any((output/name).exists() for name in names + ['plot_metadata.json', 'FIGURE_CAPTION.txt']):
        raise FileExistsError('Preserve existing curve artifacts; choose a fresh output directory')
    config = frozen['experiment_config']
    is_mpe = config.get('environment_adapter') == 'mpe_simple_spread'
    x = np.asarray([row['actual_train_env_steps'] for row in rows], dtype=float)
    if not len(x) or not np.isfinite(x).all() or (np.diff(x) <= 0).any():
        raise ValueError('Expected nonempty summaries at strictly increasing actual steps')
    panels = [('return', 'Team episode return')]
    if is_mpe:
        panels += [('final_coverage_fraction', 'Final landmark\ncoverage fraction'),
                   ('success_rate', 'Final all-landmarks\ncoverage rate'),
                   ('final_mean_min_distance', 'Final mean nearest-agent\ndistance')]
    else:
        panels += [('success_rate', 'Success rate')]
    label = {'mappo': 'MAPPO', 'soft': 'VLM4Coop (Soft)', 'hard': 'VLM4Coop (Hard)'}[frozen['method']]
    color = '#4D4D4D' if frozen['method'] == 'mappo' else '#0F4D92'
    with plt.rc_context({'font.family': ['DejaVu Sans', 'sans-serif'], 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 1.2, 'legend.frameon': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2 if is_mpe else 1, 2, figsize=(7.1, 4.8 if is_mpe else 2.65))
        try:
            for ax, (metric, ylabel) in zip(np.asarray(axes).flat, panels):
                mean = np.asarray([row[metric+'_mean'] for row in rows], dtype=float)
                sd = np.asarray([row[metric+'_std'] for row in rows], dtype=float)
                if not np.isfinite(mean).all() or not np.isfinite(sd).all() or (sd < 0).any():
                    raise ValueError(f'Invalid saved summary for {metric}')
                ax.plot(x, mean, color=color, lw=1.25, label=f'{label} (seed {frozen["seed"]})')
                ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.16, linewidth=0)
                ax.set(xlabel='Joint training environment steps', ylabel=ylabel, xlim=(0, max(x)))
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                ax.legend(fontsize=7.5, loc='best')
            fig.tight_layout(pad=1)
            for name in names:
                fig.savefig(output/name, dpi=300)
        finally:
            plt.close(fig)
    caption = (
        f'{label}, one training seed ({frozen["seed"]}). Each point uses '
        f'{config["eval_episodes_every_checkpoint"]} stochastic evaluation episodes on fixed seeds '
        f'{config["fixed_eval_seeds"]["start"]}..'
        f'{config["fixed_eval_seeds"]["start"]+config["eval_episodes_every_checkpoint"]-1}. '
        'Solid lines are episode means; shaded bands show ±1 sample SD (ddof=1), '
        'not a confidence interval or variation across training seeds. '
        'No smoothing, downsampling, fabricated initial points or intermediate evaluations; '
        'lines connect observed checkpoints. The x axis uses actual joint training steps; '
        'evaluation steps are excluded. Mean±SD bands may extend beyond metric bounds. '
        f'Evaluations are scheduled every {config["eval_interval_steps"]} training steps. '
    )
    if is_mpe:
        caption += ('MPE simple_spread: 3 agents/3 landmarks, native team reward. '
                    'All episodes last 25 steps. Final landmark coverage is geometric '
                    '(nearest agent center distance <0.1); final all-landmarks coverage is '
                    'supplementary and is not early termination or completion time. ')
    (output/'FIGURE_CAPTION.txt').write_text(caption+'\n')
    metadata = dict(files=names, points=len(rows), actual_steps=x.astype(int).tolist(),
        metrics=[metric for metric, _ in panels], mean_source='saved evaluation summaries',
        sd='sample SD over evaluation episodes (ddof=1)', smoothing=False,
        training_seed=frozen['seed'], evaluation_interval_steps=config['eval_interval_steps'],
        versions=dict(matplotlib=matplotlib.__version__, numpy=np.__version__, Pillow=pillow_version))
    (output/'plot_metadata.json').write_text(json.dumps(metadata, indent=2))
    return dict(files=names, points=len(rows), metadata='plot_metadata.json', caption='FIGURE_CAPTION.txt')
