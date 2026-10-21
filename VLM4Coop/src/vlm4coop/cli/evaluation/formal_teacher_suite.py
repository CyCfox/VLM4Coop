"""Sequential authorized teacher experiments; one configured request pool.

Never poll intermediate scores. Each training call persists full raw evidence;
only stage completion advances this queue. Credentials stay in process memory.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from vlm4coop.training.configuration import PROJECT, load_config
from vlm4coop.training.loop import train
from vlm4coop.evaluation.pure_vlm import run as run_pure
from vlm4coop.utils.provenance import capture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--capability', type=Path, required=True)
    args = parser.parse_args()
    config = deepcopy(load_config())
    assert config['protocol_status'] == 'user_resolved_2026_10_17'
    assert config['environment']['field_size'] == [8, 8]
    assert config['formal_train_steps'] == 1000000
    assert config['teacher']['concurrency'] == 1000
    assert config['teacher']['top_p'] == .95
    assert config['teacher']['thinking'] == {'type': 'disabled'}
    assert 'reasoning_effort' not in config['teacher']
    assert config['teacher']['pure_vlm_use_cache'] is False
    smoke = json.loads((args.smoke / 'status.json').read_text())
    assert smoke['status'] == 'passed' and smoke['selected_debug_labels'] == 100
    assert 2 <= smoke['max_actual_concurrent_requests'] <= 100
    assert smoke['student_eval_episodes'] == 100 and smoke['api_requests_during_ppo_and_eval'] == 0
    smoke_config = json.loads((args.smoke / 'config.json').read_text())['experiment_config']
    assert smoke_config['teacher'] == config['teacher']
    assert smoke_config['environment'] == config['environment']
    capability = json.loads((args.capability / 'summary.json').read_text())
    capability_config = json.loads((args.capability / 'config.json').read_text())
    assert capability_config['environment'] == config['environment']
    for key in ('endpoint', 'model', 'thinking', 'reasoning_effort', 'prompt_version', 'top_p'):
        assert capability_config.get(key) == config['teacher'].get(key)
    assert capability['status'] == 'complete' and capability['connectivity']['success']
    assert capability['perception']['agents'] and capability['perception']['food']
    assert any(row['success'] for row in capability['episodes']), 'No successful closed-loop sanity episode'
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT.resolve()):
        raise ValueError('Suite output must remain under VLM4Coop')
    output.mkdir(parents=True, exist_ok=False)
    capture(output)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    mapping = {'mappo_3': 'VLM4Coop/results/mappo_8x8_formal_fixed500_1000k_20261017_01'}
    status = {'status': 'running', 'completed': [], 'current': None}
    def write():
        (output / 'status.json').write_text(json.dumps(status, indent=2))
        (output / 'runs.json').write_text(json.dumps(mapping, indent=2))
    try:
        for method, agents in [('soft', 3), ('hard', 3), ('soft', 4), ('soft', 5)]:
            key = f'{method}_{agents}'
            status['current'] = key
            write()
            print(json.dumps({'stage': key, 'event': 'started'}), flush=True)
            destination = output / key
            train(method, agents, 0, 'formal', destination, config=config)
            mapping[key] = str(destination.relative_to(PROJECT.parent))
            status['completed'].append(key)
            write()
            print(json.dumps({'stage': key, 'event': 'complete'}), flush=True)
        status['current'] = 'pure_vlm_3'
        write()
        destination = output / 'pure_vlm_3'
        run_pure(destination, config=config)
        mapping['pure_vlm_3'] = str(destination.relative_to(PROJECT.parent))
        status.update(status='complete', current=None)
        status['completed'].append('pure_vlm_3')
        write()
        print('FORMAL_TEACHER_SUITE_COMPLETE', flush=True)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        write()
        raise


if __name__ == '__main__':
    main()
