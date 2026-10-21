"""One genuine new-state request before restarting the authorized suite."""
import json
from pathlib import Path
from PIL import Image
from vlm4coop.training.configuration import load_config, PROJECT
from vlm4coop.envs.lbf.adapter import register_lbf, CoopGymmaWrapper
from vlm4coop.teachers.factory import make_teacher

config = load_config()
output = PROJECT / 'results/go_recovery_health_20261017_01'
output.mkdir(exist_ok=False)
key = register_lbf(3, config['environment'])
env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None, seed=0,
                      common_reward=True, reward_scalarisation='sum')
teacher = None
try:
    env.reset(seed=912347)
    image = Image.fromarray(env.render())
    image.save(output / 'input.png')
    teacher = make_teacher(config, output)
    result = teacher.predict(image, 3)
    assert result.probs.shape == (3, 6)
    calls = teacher.calls
    assert teacher.predict(image, 3).cache_hit and teacher.calls == calls
    status = dict(status='passed', formal=False, teacher_calls=calls,
                  cache_hit_after_request=True, endpoint=config['teacher']['endpoint'])
    (output / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
except BaseException as exc:
    (output / 'status.json').write_text(json.dumps(dict(status='failed', error=repr(exc))))
    raise
finally:
    env.close()
    if teacher is not None: teacher.cache.close()
