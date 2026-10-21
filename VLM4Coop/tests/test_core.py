import json
import math
import numpy as np
import pytest
import torch
from PIL import Image
from vlm4coop.algorithms.distillation import kd_weight, distillation_loss
from vlm4coop.evaluation.metrics import summarize
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.teachers.interfaces import TeacherResult
from vlm4coop.teachers.legacy_parser import parse_probabilities


def raw(rows):
    return json.dumps({"agents": [{"id": i, "probs": p} for i, p in rows]})


def test_kl_and_hard_gradients_and_padding():
    logits = torch.zeros(1, 2, 2, 6, requires_grad=True)
    p = torch.zeros_like(logits)
    p[..., 0] = 0.8
    p[..., 1] = 0.2
    p[:, 1] = float("nan")  # invalid padded labels must be ignored
    mask = torch.tensor([[[1., 1.], [0., 0.]]])
    soft = distillation_loss(logits.softmax(-1), p, mask, "soft")
    expected = 0.8 * math.log(0.8 * 6) + 0.2 * math.log(0.2 * 6)
    assert soft.item() == pytest.approx(expected)
    soft.backward()
    assert torch.allclose(logits.grad[0, 0, 0], (torch.full((6,), 1/6) - p[0, 0, 0]) / 2)
    assert logits.grad[:, 1].abs().sum() == 0
    hard = distillation_loss(logits.softmax(-1), p, mask, "hard")
    assert hard.item() == pytest.approx(math.log(6))
    assert not p.requires_grad


def test_schedule_boundaries():
    assert kd_weight(0) == 10
    assert kd_weight(33000) == pytest.approx(5.05)
    assert kd_weight(66000) == .1
    assert kd_weight(99999) == .1
    assert kd_weight(100000) == 0


def test_parser_sort_and_thinking():
    text = '<think>draft {"agents": []}</think>\n```json\n' + raw([(1, [0, 0, 0, 0, 0, 1]), (0, [1, 0, 0, 0, 0, 0])]) + '\n```'
    p, repaired = parse_probabilities(text, 2)
    assert not repaired
    assert p.argmax(-1).tolist() == [0, 5]


def test_parser_uses_final_json_after_visible_reasoning_and_draft():
    draft = raw([(0, [1, 0, 0, 0, 0, 0])])
    final = raw([(0, [0, 0, 0, 0, 0, 1])])
    text = 'Positions: {not JSON}. Preliminary draft: ' + draft + '\nCheck adjacency; LOAD together.\n```json\n' + final + '\n```'
    p, repaired = parse_probabilities(text, 1)
    assert p.argmax(-1).tolist() == [5] and not repaired


def test_parser_recovers_observed_missing_agents_list_bracket_without_new_values():
    text = 'Brief reasoning.\nified JSON:\n{"agents":[{"id":0,"probs":[0,0,0,0,0,1]},{"id":1,"probs":[0,0,0,0,1,0]},{"id":2,"probs":[0,0,0,0,0,1]}}'
    p, repaired = parse_probabilities(text, 3)
    assert repaired
    np.testing.assert_array_equal(p, [[0,0,0,0,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]])


def test_incomplete_final_rows_do_not_fall_back_to_a_reasoning_draft():
    draft = raw([(0,[1,0,0,0,0,0])])
    with pytest.raises(ValueError, match='Incomplete agent'):
        parse_probabilities('Draft: '+draft+'\nFinal: {"agents":[{"id":0,"probs":[0,0,0', 1)


def test_observed_seven_values_preserves_identifiable_load_maximum():
    p,repaired=parse_probabilities(raw([(0,[0,0,0,0,0,.98,.02])]),1)
    assert repaired and p.shape==(1,6)
    assert p[0,5]==pytest.approx(.98)
    np.testing.assert_allclose(p[0,:5],np.full(5,.004))
    assert p.sum()==pytest.approx(1)


def test_extra_entry_with_maximum_outside_six_actions_cannot_be_invented():
    with pytest.raises(ValueError,match='no legal action index'):
        parse_probabilities(raw([(0,[0,0,0,0,0,0,1])]),1)


def test_parser_repair_preserves_maximum_probability():
    p, repaired = parse_probabilities(raw([(0, [0.1, 0.8, 0.1, 0.1, 0.1, 0.1])]), 1)
    assert repaired
    assert p[0, 1] == pytest.approx(.8)
    assert p[0].sum() == pytest.approx(1)
    assert p[0, 0] == pytest.approx(.04)


@pytest.mark.parametrize("text,n", [(raw([(0, [1, 0, 0, 0, 0, 0]), (0, [1, 0, 0, 0, 0, 0])]), 2), ('garbage', 1), (raw([(0, [2, 0, 0, 0, 0, 0])]), 1)])
def test_parser_rejects_unrecoverable(text, n):
    with pytest.raises(ValueError):
        parse_probabilities(text, n)


def test_cache_persists_and_separates_identity_and_shape(tmp_path):
    image = Image.new("RGB", (5, 5))
    cache = ProbabilityCache(tmp_path / 'teacher.sqlite', {"model": "a", "prompt": "v1"})
    key = cache.key(image, 1)
    result = TeacherResult(np.array([[1, 0, 0, 0, 0, 0]], np.float32), 'real stored reply', False)
    assert cache.get(key) is None
    assert cache.put(key, result)
    cache.close()
    cache = ProbabilityCache(tmp_path / 'teacher.sqlite', {"model": "a", "prompt": "v1"})
    assert cache.get(key).cache_hit
    assert cache.key(image, 2) != key
    cache.close()
    other = ProbabilityCache(tmp_path / 'teacher.sqlite', {"model": "a", "prompt": "v2"})
    assert other.get(other.key(image, 1)) is None
    other.close()


def test_cache_budget_stops_new_entries(tmp_path):
    cache = ProbabilityCache(tmp_path / 'tiny.sqlite', {}, max_bytes=8192)
    assert not cache.put('key', TeacherResult(np.ones((1, 6))/6, '{}', False))
    assert cache.get('key') is None
    cache.close()


def test_success_length_excludes_failures():
    result = summarize([dict(success=True, team_return=1., length=12),
                        dict(success=False, team_return=.5, length=100)])
    assert result['success_rate_mean'] == .5
    assert result['success_episodes_length_mean'] == 12
    assert math.isnan(result['success_episodes_length_std'])
    assert result['episodes_length_mean'] == 56
    result = summarize([dict(success=False, team_return=0., length=100)])
    assert math.isnan(result['success_episodes_length_mean'])


def test_parser_repairs_negative_entries_preserving_identifiable_maximum():
    p, repaired = parse_probabilities(raw([(0, [.8, -.1, .1, .1, .1, 0])]), 1)
    assert repaired
    assert p[0, 0] == pytest.approx(.8)
    assert np.allclose(p[0, 1:], .04)
