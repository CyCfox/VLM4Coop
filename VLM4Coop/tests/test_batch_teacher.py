"""Caching/order plumbing only; genuine model batching has a separate GPU probe."""
from pathlib import Path
from PIL import Image
import numpy as np
from vlm4coop.teachers.batched_local import BatchedQwenTeacher
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.teachers.interfaces import TeacherResult


def test_cached_batch_order_and_duplicate_state_alignment(tmp_path):
    teacher = object.__new__(BatchedQwenTeacher)
    teacher.config = {'batch_size': 8}
    teacher.cache = ProbabilityCache(tmp_path / 'cache.sqlite', {'protocol': 'test'})
    images = [Image.new('RGB', (4, 4), color) for color in ('red', 'green')]
    a = np.zeros((3, 6), np.float32)
    b = a.copy()
    a[:, 1], b[:, 4] = 1, 1
    teacher.cache.put(teacher.cache.key(images[0], 3), TeacherResult(a, '{}', False))
    teacher.cache.put(teacher.cache.key(images[1], 3), TeacherResult(b, '{}', False))
    # No processor/model exists: a correct all-hit path must not touch inference.
    result = teacher.predict_many([images[1], images[0], images[1]], 3)
    assert [item.probs.argmax(-1).tolist() for item in result] == [[4]*3, [1]*3, [4]*3]
    teacher.cache.close()
