"""Bounded-thinking generation primitive, enabled by user authorization.

The user approved a uniform 256-token thinking budget on 2026-10-17.
The processor lets the model produce genuine reasoning tokens, then emits the
model's own closing-think token within the same generation. It never supplies
an action or a probability. Teacher thinking mode remains enabled.
"""
import torch


class ThinkingBudget:
    def __init__(self, prompt_length, closing_token_id, budget):
        if budget <= 0:
            raise ValueError('Thinking budget must be positive')
        self.prompt_length = prompt_length
        self.closing_token_id = closing_token_id
        self.budget = budget

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] < self.budget:
            return scores
        completed = (generated == self.closing_token_id).any(dim=-1)
        must_close = ~completed
        if must_close.any():
            scores = scores.clone()
            scores[must_close] = -torch.inf
            scores[must_close, self.closing_token_id] = 0
        return scores
