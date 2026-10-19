from pathlib import Path
import yaml
p=Path('VLM4Coop/src/vlm4coop/api_teacher.py')
s=p.read_text().replace("for name in ('thinking', 'reasoning_effort'):", "for name in ('thinking', 'reasoning_effort', 'temperature'):")
s=s.replace("'reasoning_effort': payload.get('reasoning_effort', 'provider_default'),", "'reasoning_effort': payload.get('reasoning_effort', 'provider_default'),\n                 'temperature_requested': payload.get('temperature'),")
s=s.replace('Agents cannot move outside the grid, onto food, or into occupied cells.', 'Agents cannot move outside the grid or onto food. If agents target the same cell their moves fail; swaps or moves into simultaneously vacated cells are allowed.')
p.write_text(s)
for name in ('api_probe.yaml', 'experiments.yaml'):
    p=Path('VLM4Coop/configs') / name
    data=yaml.safe_load(p.read_text())
    section=data['teacher'] if name=='experiments.yaml' else data
    section['prompt_version']='joint-lbf-api-v3-native-movement'
    p.write_text(yaml.safe_dump(data,sort_keys=False))
