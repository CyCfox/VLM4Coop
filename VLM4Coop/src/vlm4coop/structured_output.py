"""Constrain JSON shape after thinking; all numeric values are model-selected.

Uses LM Format Enforcer's tokenizer tree/regex engine directly because its
0.11.3 transformers integration imports a tokenizer class removed in v5.
No environment state, action legality or joint-action consistency is consulted.
"""
import torch
from lmformatenforcer import TokenEnforcer, TokenEnforcerTokenizerData, RegexParser


def build_tokenizer_data(tokenizer):
    special = set(tokenizer.all_special_ids)
    prefix = tokenizer.encode('0', add_special_tokens=False)[-1]
    entries = []
    for token in range(len(tokenizer)):
        if token in special:
            continue
        plain = tokenizer.decode([token], clean_up_tokenization_spaces=False)
        joined = tokenizer.decode([prefix, token], clean_up_tokenization_spaces=False)[1:]
        if joined:
            entries.append((token, joined, len(joined) > len(plain)))
    def decode(tokens):
        return tokenizer.decode(tokens, clean_up_tokenization_spaces=False).rstrip('\ufffd')
    return TokenEnforcerTokenizerData(entries, decode, tokenizer.eos_token_id, False, len(tokenizer))


def matrix_pattern(n_agents, decimals=3):
    if n_agents not in (3, 4, 5) or decimals < 1:
        raise ValueError('Unsupported probability matrix schema')
    probability = rf'(0(\.[0-9]{{1,{decimals}}})?|1(\.0{{1,{decimals}}})?)'
    rows = [r'\{"id":' + str(i) + r',"probs":\[' + ','.join([probability]*6) + r'\]\}'
            for i in range(n_agents)]
    return r'[ \n]{0,2}\{"agents":\[' + ','.join(rows) + r'\]\}[ \n]{0,2}'


class MatrixJsonOutput:
    def __init__(self, tokenizer_data, prompt_length, closing_token_id, n_agents, decimals=3):
        self.prompt_length, self.closing_token_id = prompt_length, closing_token_id
        self.enforcer = TokenEnforcer(tokenizer_data, RegexParser(matrix_pattern(n_agents, decimals)))
        eos = tokenizer_data.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    def __call__(self, input_ids, scores):
        filtered = scores.clone()
        for row, ids in enumerate(input_ids[:, self.prompt_length:].tolist()):
            if self.closing_token_id not in ids:
                continue  # free model-generated thinking, independently bounded
            answer = ids[ids.index(self.closing_token_id)+1:]
            if any(token in self.eos for token in answer):
                continue  # finished row waiting for other batch samples
            allowed = self.enforcer.get_allowed_tokens(answer).allowed_tokens
            if not allowed:
                raise ValueError('JSON format grammar has no continuation')
            filtered[row] = -torch.inf
            filtered[row, allowed] = scores[row, allowed]
        return filtered
