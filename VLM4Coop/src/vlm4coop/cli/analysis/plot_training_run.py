"""Backfill curves for a completed formal run into a fresh directory, without training."""
import argparse
import json
from pathlib import Path
from vlm4coop.evaluation.analysis import load_run
from vlm4coop.evaluation.plotting import plot_training_curves
from vlm4coop.utils.paths import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not all(p.is_relative_to(PROJECT.resolve()) for p in (run, output)):
        raise ValueError('Run and output must stay inside VLM4Coop')
    frozen = json.loads((run/'config.json').read_text())
    if frozen['method'] not in ('mappo', 'soft', 'hard'):
        raise ValueError('Pure VLM has no training curve')
    data = load_run(run, frozen['method'], frozen['agents'])
    output.mkdir(parents=True, exist_ok=False)
    result = plot_training_curves(data['evaluations'], frozen, output)
    result.update(source_run=str(run), new_training=False, API_requests=0)
    (output/'source.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
