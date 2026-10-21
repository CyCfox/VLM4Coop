"""At most six bounded real MPE requests; no rollout or training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from PIL import Image
from vlm4coop.envs.mpe.adapter import MPEGymBridge
from vlm4coop.envs.registry import get_adapter
from vlm4coop.training.configuration import load_config
from vlm4coop.teachers.api import OpenCodeTeacher, off_response_evidence
from vlm4coop.teachers.response_protocol import parse_api_probabilities
from vlm4coop.utils.paths import PROJECT


def run(output, control='configured', requests=6, provider_budget=False):
    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT.resolve()):
        raise ValueError('Project-owned output required')
    output.mkdir(parents=True, exist_ok=False)
    config = deepcopy(load_config(PROJECT/'configs/mpe_spread.yaml'))
    config['teacher']['max_tokens'] = None if provider_budget else 1024
    if control == 'none':
        config['teacher'].pop('chat_template_kwargs', None)
        config['teacher']['reasoning_effort'] = 'none'
    elif control == 'template-off':
        config['teacher'].pop('reasoning_effort', None)
        config['teacher']['chat_template_kwargs'] = dict(thinking=False, enable_thinking=False)
    elif control == 'native-low':
        config['teacher']['reasoning_effort'] = 'low'
    elif control == 'reasoning-off':
        config['teacher'].pop('reasoning_effort', None)
        config['teacher']['reasoning'] = dict(enabled=False, effort='none')
    elif control == 'enable-off':
        config['teacher'].pop('reasoning_effort', None)
        config['teacher']['enable_thinking'] = False
    (output/'config.json').write_text(json.dumps(config, indent=2))
    spec = get_adapter(config).teacher_spec()
    teacher = OpenCodeTeacher(config['teacher'], output/'teacher_calls.jsonl', purpose='pure', protocol=spec)
    teacher.transport_policy['max_attempts'] = 1
    rows = []
    began = time.monotonic()
    env = MPEGymBridge()
    status = 'failed'
    error = None
    try:
        for i in range(requests):
            env.reset(seed=1000000+i%3)
            # Second set verifies the same controls on genuine moving states.
            if i >= 3:
                for _ in range(5): env.step([1,2,3])
            image = Image.fromarray(env.render())
            image.save(output/f'frame_{i}.png')
            raw, response = teacher.request(spec.user_prompt(3), image=image,
                system=spec.system_prompt(3), label=f'OFF_control_{i}')
            probs, repaired = parse_api_probabilities(raw,3,'pure',5)
            rows.append(dict(index=i, image_size=image.size, evidence=off_response_evidence(response),
                usage=response.get('usage'), probabilities=probs.tolist(), repaired=repaired,
                finish_reason=response['choices'][0].get('finish_reason')))
        status = 'passed' if all(r['evidence'] != 'returned_reasoning' for r in rows) else 'failed'
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
        result = dict(status=status, formal=False, real_HTTP_calls=teacher.calls,
            completed_labels=len(rows), responses_with_reasoning=teacher.responses_with_reasoning,
            wall_seconds=time.monotonic()-began, error=error, rows=rows,
            note='Small control test of observed output only; no provider-wide guarantee or capability claim.')
        (output/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--control', choices=['configured','none','template-off','native-low','reasoning-off','enable-off'], default='configured')
    parser.add_argument('--requests',type=int,choices=[3,6],default=6)
    parser.add_argument('--provider-budget',action='store_true',help='Use the unchanged formal max_tokens=None setting')
    args = parser.parse_args()
    raise SystemExit(run(args.output,args.control,args.requests,args.provider_budget))
