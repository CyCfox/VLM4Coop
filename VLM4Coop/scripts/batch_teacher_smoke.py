"""Small real throughput probe; training does not use this class until verified."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time
import torch
from PIL import Image
from vlm4coop.config import load_config, PROJECT
from vlm4coop.environment import register_lbf, CoopGymmaWrapper
from vlm4coop.training import make_teacher
from vlm4coop.batched_teacher import BatchedQwenTeacher

config = load_config()
config['teacher']['batch_size'] = 8
output = PROJECT / 'results' / ('teacher_batch_smoke_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
output.mkdir(exist_ok=False)
print(json.dumps({'output': str(output)}), flush=True)
# Same loading and configuration as the validated single-image teacher.
teacher = make_teacher(config, output, teacher_type=BatchedQwenTeacher)
key = register_lbf(3, config['environment'])
env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None, seed=0,
                      common_reward=True, reward_scalarisation='sum')
try:
    images = []
    for seed in range(8):
        env.reset(seed=1200 + seed)
        images.append(Image.fromarray(env.render()))
    start = time.monotonic()
    results = teacher.predict_many(images, 3, use_cache=False)
    elapsed = time.monotonic() - start
    result = dict(status='passed', samples=len(results), batch_size=8, wall_seconds=elapsed,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(),
                  generated_state_outputs=teacher.calls, physical_generate_calls=teacher.generate_calls,
                  probs=[item.probs.tolist() for item in results])
    (output / 'measurements.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
finally:
    env.close()
    teacher.cache.close()
