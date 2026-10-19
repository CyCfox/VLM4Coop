"""SYNTHETIC teacher only: exercise the real 100-worker Pure VLM path, no API."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import threading
import numpy as np
from vlm4coop.api_batch import predict_many
from vlm4coop.config import load_config
from vlm4coop.pure_vlm import run
from vlm4coop.teacher import TeacherResult


class SyntheticTeacher:
    """No HTTP: emulate three failed formats, exercising the NONE fallback."""
    def __init__(self, config):
        self.config = config['teacher']
        self.cache = None
        self.calls = 0
        self.max_active_requests = None
        self._lock = threading.Lock()
        self.format_failures=self.format_retries=self.failed_decisions=0

    def predict(self, image, n_agents, use_cache):
        assert not use_cache and n_agents == 3
        with self._lock:
            self.calls += 3
            self.format_failures+=3; self.format_retries+=2; self.failed_decisions+=1
            decision_id=f'synthetic-{self.failed_decisions}'
        return TeacherResult(None, 'SYNTHETIC SMOKE ONLY; THREE FAILED FORMATS; NO API', False,
            source='format_failure',format_attempts=3,decision_id=decision_id)

    def predict_many(self, images, n_agents, use_cache):
        return predict_many(self, images, n_agents, use_cache)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = deepcopy(load_config())
    assert config['pure_vlm_parallel_envs'] == config['pure_vlm_concurrency'] == 100
    assert not config['teacher']['pure_vlm_use_cache']
    # Clearly mark before any execution, including the frozen source/config.
    config['test_double'] = 'SYNTHETIC_SMOKE_ONLY_NOT_FORMAL_RESULTS'
    teacher = SyntheticTeacher(config)
    def factory(actual_config, output, use_cache, purpose):
        assert use_cache is False
        assert purpose=='pure'
        teacher.config = actual_config['teacher']
        return teacher
    with patch('vlm4coop.pure_vlm.make_teacher', factory):
        status = run(a.output, config=config, profile='smoke')
    episodes = [json.loads(r) for r in (a.output/'eval_episodes.jsonl').read_text().splitlines()]
    steps = [json.loads(r) for r in (a.output/'pure_vlm_steps.jsonl').read_text().splitlines()]
    assert status['n_eval']==100 and len(episodes)==100
    assert [r['env_seed'] for r in episodes]==list(range(1000000,1000100))
    assert all(r['length']==100 and not r['success'] for r in episodes)
    assert len(steps)==status['decisions']==status['none_fallback_decisions']==10000
    assert status['teacher_calls']==30000 and status['format_retries']==20000
    assert all(row['source']=='pure_none_fallback' and row['actions']==[0,0,0] for row in steps)
    assert not any(r['cache_hit'] for r in steps)
    assert len(list((a.output/'frames').glob('*.png')))==10000
    status.update(profile='synthetic_transport_smoke_only', real_http_requests=0,
                  mock_teacher_decisions=10000,mock_format_attempts=30000,
                  verified_fixed_environment_seeds=True)
    (a.output/'SMOKE_VALIDATION.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status))


if __name__ == '__main__':
    main()
