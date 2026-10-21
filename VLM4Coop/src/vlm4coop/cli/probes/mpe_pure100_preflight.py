"""100 real MPE workers x25 steps, synthetic labels only; no API credentials."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from vlm4coop.training.configuration import load_config
from vlm4coop.utils.paths import PROJECT
from vlm4coop.envs.registry import get_adapter
from vlm4coop.evaluation import pure_vlm
from vlm4coop.evaluation.mpe_replay import replay_pure_mpe
from vlm4coop.teachers.interfaces import TeacherResult


def run(output):
    config=deepcopy(load_config(PROJECT/'configs/mpe_pure_vlm100.yaml'))
    config['test_double']='ENGINEERING ONLY: synthetic teacher labels; no API or model ability'
    spec=get_adapter(config).teacher_spec()
    class Synthetic:
        calls=0
        cache=None
        protocol=spec
        batches=0
        config={'pure_vlm_use_cache':False}
        def predict_many(self,images,n_agents,use_cache):
            if not images:
                return []  # Native runner requests a terminal action without a frame.
            assert len(images)==100 and n_agents==3 and use_cache is False
            self.batches+=1
            p=np.tile([1,0,0,0,0],(3,1)).astype(np.float32)
            return [TeacherResult(p,'SYNTHETIC TEST ONLY',False) for _ in images]
    teacher=Synthetic()
    with patch.object(pure_vlm,'make_teacher',return_value=teacher):
        status=pure_vlm.run(output,config=config,profile='smoke')
    output=Path(output)
    episodes=[json.loads(r) for r in (output/'eval_episodes.jsonl').read_text().splitlines()]
    steps=[json.loads(r) for r in (output/'pure_vlm_steps.jsonl').read_text().splitlines()]
    assert status['n_eval']==100 and status['parallel_envs']==100 and status['decisions']==2500
    assert teacher.calls==0 and teacher.batches==25 and len(steps)==2500
    assert [e['env_seed'] for e in episodes]==list(range(1000000,1000100))
    assert len(list((output/'frames').glob('*.png')))==2500
    replay=replay_pure_mpe(output,config,episodes,steps)
    result=dict(status='passed',engineering_only=True,real_API_calls=0,workers=100,
        episodes=100,steps=2500,batches=25,native_RGB_replay=replay)
    (output/'engineering_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
