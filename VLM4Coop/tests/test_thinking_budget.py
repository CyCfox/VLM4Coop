"""Mechanism checks only: the user-authorized primitive never supplies actions/probabilities."""
import torch
from vlm4coop.teachers.thinking_budget import ThinkingBudget


def test_budget_does_not_force_actions_or_change_completed_reasoning():
    processor = ThinkingBudget(prompt_length=2, closing_token_id=7, budget=3)
    scores = torch.randn(2, 12)
    before = torch.tensor([[1, 2, 3, 4], [1, 2, 7, 5]])
    assert torch.equal(processor(before, scores), scores)
    after = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 7, 4, 5]])
    output = processor(after, scores)
    assert output[0].argmax().item() == 7
    assert torch.isneginf(output[0, torch.arange(12) != 7]).all()
    assert torch.equal(output[1], scores[1])
    assert torch.isfinite(scores).all()  # original scores are unchanged
