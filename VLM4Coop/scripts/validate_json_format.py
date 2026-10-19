"""Actual Qwen tokenizer / format engine checks; no model inference or results."""
import json
import torch
from transformers import AutoTokenizer
from vlm4coop.config import PROJECT
from vlm4coop.structured_output import build_tokenizer_data, MatrixJsonOutput
from vlm4coop.teacher import parse_probabilities

tokenizer = AutoTokenizer.from_pretrained('/workspace/models/Qwen3.5-4B', local_files_only=True)
data = build_tokenizer_data(tokenizer)
closing = tokenizer.encode('</think>', add_special_tokens=False)[0]
for n in (3, 4, 5):
    text = json.dumps({'agents': [{'id': i, 'probs': [.05, .15, .1, .1, .5, .1]} for i in range(n)]}, separators=(',', ':'))
    target = tokenizer.encode(text, add_special_tokens=False)
    guide = MatrixJsonOutput(data, 0, closing, n, 3)
    ids = [closing]
    for token in target:
        scores = torch.zeros(1, len(tokenizer))
        filtered = guide(torch.tensor([ids]), scores)
        assert torch.isfinite(filtered[0, token]), (n, tokenizer.decode(ids), tokenizer.decode([token]))
        ids.append(token)
    filtered = guide(torch.tensor([ids]), torch.zeros(1, len(tokenizer)))
    assert torch.isfinite(filtered[0, tokenizer.eos_token_id])
    probs, repaired = parse_probabilities(tokenizer.decode(ids), n)
    assert probs.shape == (n, 6) and not repaired
print(json.dumps({'status': 'passed', 'matrix_schemas': [3, 4, 5], 'actual_tokenizer': True}))
