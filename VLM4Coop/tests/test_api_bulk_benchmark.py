"""Offline benchmark accounting and durable output, without real API calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from vlm4coop.teachers.api_batch import predict_many
from vlm4coop.teachers.api import build_api_cache_identity
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.teachers.interfaces import TeacherResult


def load_benchmark():
    from vlm4coop.cli.probes import api_bulk_benchmark as module
    return module


@pytest.mark.parametrize('fail', [False, True])
def test_single_batch_timing_and_partial_cache(tmp_path, fail):
    config = dict(concurrency=1000, top_p=.95, model='synthetic-test')
    cache = ProbabilityCache(tmp_path / 'test.sqlite', build_api_cache_identity(config))
    teacher = SimpleNamespace(config=config, cache=cache, calls=0, max_active_requests=0)
    submitted = []
    def predict(image, n_agents, use_cache):
        teacher.calls += 1
        teacher.max_active_requests = 1
        if fail and image.getpixel((0, 0))[0] == 1:
            raise RuntimeError('synthetic test failure')
        return TeacherResult(np.full((3, 6), 1 / 6, dtype=np.float32), 'test only', False)
    def bulk(images, n_agents, use_cache):
        submitted.append(len(images))
        return predict_many(teacher, images, n_agents, use_cache)
    teacher.predict, teacher.predict_many = predict, bulk
    images = [Image.new('RGB', (2, 2), (i, 0, 0)) for i in (0, 1, 0)]
    positions = [(0, 0), (1, 0), (0, 1)]
    try:
        if fail:
            with pytest.raises(RuntimeError, match='synthetic test failure'):
                load_benchmark().label_batch(teacher, images, positions,
                                            [{'env_seed': 10}, {'env_seed': 20}], tmp_path)
        else:
            load_benchmark().label_batch(teacher, images, positions,
                                        [{'env_seed': 10}, {'env_seed': 20}], tmp_path)
        timing = json.loads((tmp_path / 'timing.json').read_text())
        assert submitted == [3] and timing['batch_calls'] == 1
        assert timing['input_states'] == 3 and timing['unique_states'] == 2
        assert timing['http_attempts'] == 2 and timing['batch_wall_seconds'] > 0
        assert timing['status'] == ('failed' if fail else 'complete')
        assert timing['cache_writes'] == (1 if fail else 2)
        if not fail:
            assert timing['persistent_cache_verification'] == 'passed'
            assert len((tmp_path / 'teacher_targets.jsonl').read_text().splitlines()) == 3
    finally:
        cache.close()
