from pathlib import Path
import yaml
for name in ('api_probe.yaml', 'experiments.yaml'):
    p=Path('VLM4Coop/configs') / name
    data=yaml.safe_load(p.read_text())
    section=data['teacher'] if name=='experiments.yaml' else data
    section['thinking']={'type':'enabled'}
    section['reasoning_effort']='low'
    section['max_tokens']=None
    section.pop('temperature',None)
    if name=='experiments.yaml':
        data['protocol_status']='api_teacher_validation_pending'
    p.write_text(yaml.safe_dump(data,sort_keys=False))
p=Path('VLM4Coop/scripts/api_teacher_probe.py')
s=p.read_text()
s=s.replace("dump(output / 'config.json', dict(config, episodes=args.episodes,", "dump(output / 'config.json', dict(config, episodes=args.episodes,\n         environment=load_config()['environment'], input_mode='rgb_only', provided_coordinates=False,")
s=s.replace("    print(json.dumps({'output': str(output)}), flush=True)", "    from vlm4coop.provenance import capture\n    capture(output)\n    print(json.dumps({'output': str(output)}), flush=True)")
s=s.replace("'formal': False, 'model': config['model']}", "'formal': False, 'model': config['model'],\n              'input_mode': 'rgb_only', 'provided_coordinates': False,\n              'thinking': config.get('thinking'), 'reasoning_effort': config.get('reasoning_effort', 'provider_default')}\n    dump(output / 'summary.json', report)")
s=s.replace("                if terminated or truncated:\n", "                dump(output / 'progress.json', dict(seed=seed, step=t+1,\n                     remaining_food=int(info['remaining_food']), api_calls=teacher.calls))\n                if terminated or truncated:\n")
p.write_text(s)
