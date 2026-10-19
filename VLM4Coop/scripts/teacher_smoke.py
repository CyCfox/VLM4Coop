"""Real local GPU teacher validation only; no quality claim from these samples."""
from pathlib import Path
import json
import time
from datetime import datetime, timezone, timedelta
import torch
import numpy as np
from PIL import Image
from vlm4coop.config import PROJECT, load_config
from vlm4coop.environment import register_lbf, CoopGymmaWrapper
from vlm4coop.training import make_teacher

output = PROJECT / 'results' / ('teacher_smoke_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
output.mkdir(parents=True, exist_ok=False)
config = load_config()
assert torch.cuda.is_available(), 'GPU unavailable'
print(json.dumps({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'output': str(output)}), flush=True)
start = time.monotonic()
teacher = make_teacher(config, output)
load_seconds = time.monotonic() - start
results = []
try:
    for n in (3, 4, 5):
        key = register_lbf(n, config['environment'])
        env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None,
                              seed=0, common_reward=True, reward_scalarisation='sum')
        try:
            env.reset(seed=1234)
            image = Image.fromarray(env.render())
            image.save(output / f'lbf_{n}p.png')
            result = teacher.predict(image, n, use_cache=False)
            assert result.probs.shape == (n, 6)
            assert np.allclose(result.probs.sum(-1), 1)
            record = {'agents': n, 'latency_seconds': result.latency_seconds,
                      'generated_tokens': result.generated_tokens, 'repaired': result.repaired,
                      'probs': result.probs.tolist()}
            results.append(record)
            (output / 'measurements.json').write_text(json.dumps({'load_seconds': load_seconds,
                'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'samples': results}, indent=2))
            print(json.dumps(record), flush=True)
        finally:
            env.close()
except BaseException as exc:
    (output / 'status.json').write_text(json.dumps({'status': 'failed', 'error': repr(exc)}))
    raise
finally:
    teacher.cache.close()
(output / 'status.json').write_text(json.dumps({'status': 'passed', 'real_teacher_calls': teacher.calls}))
print(json.dumps({'status': 'passed', 'real_teacher_calls': teacher.calls}), flush=True)
