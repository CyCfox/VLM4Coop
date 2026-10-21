"""Sequential formal job launcher. It records status, never polls learning scores."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import sys
from vlm4coop.training.configuration import PROJECT, load_config


def run(root):
    root = Path(root).resolve()
    if not root.is_relative_to(PROJECT.resolve()):
        raise ValueError('Formal suite must stay in VLM4Coop')
    root.mkdir(parents=True, exist_ok=False)
    config = load_config()
    jobs = [('mappo_3', 'mappo', 3), ('soft_3', 'soft', 3), ('hard_3', 'hard', 3),
            ('soft_4', 'soft', 4), ('soft_5', 'soft', 5), ('pure_vlm_3', 'pure_vlm', 3)]
    mapping = {key: str(root / key) for key, _, _ in jobs}
    (root / 'runs.json').write_text(json.dumps(mapping, indent=2))
    state = {'status': 'running', 'started_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'experiment_config': config, 'jobs': {key: 'pending' for key, _, _ in jobs}}
    def save():
        (root / 'suite_status.json').write_text(json.dumps(state, indent=2))
    save()
    for key, method, agents in jobs:
        state['jobs'][key] = 'running'
        save()
        command = [sys.executable, '-m', 'vlm4coop.evaluation.pure_vlm', '--output', mapping[key]] if method == 'pure_vlm' else [
            sys.executable, '-m', 'vlm4coop.training.loop', '--method', method, '--agents', str(agents),
            '--seed', str(config['train_seeds'][0]), '--profile', 'formal', '--output', mapping[key]]
        if load_config() != config:
            raise ValueError('Configuration changed during formal suite; do not mix protocols')
        with (root / f'{key}.log').open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        state['jobs'][key] = 'complete' if code == 0 else 'failed'
        state['last_return_code'] = code
        save()
        if code:
            state['status'] = 'failed'
            save()
            raise RuntimeError(f'{key} failed with code {code}; read preserved log before any retry')
    state['status'] = 'complete'
    state['completed_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root)))
