"""Strict API schema, role-specific numeric handling, and explicit fallbacks."""
import json
import re
import numpy as np


class ProbabilityFormatError(ValueError):
    pass


def clip_and_normalize(values):
    """Clip every entry to [0,1], then normalize the clipped row."""
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ProbabilityFormatError('Expected finite numeric probabilities')
    clipped = np.clip(p,0,1)
    total = clipped.sum()
    if total <= 0:
        raise ProbabilityFormatError('Clipped row has zero mass; cannot normalize')
    corrected = clipped / total
    return corrected, bool(not np.array_equal(p,clipped) or not np.isclose(total,1,atol=1e-6,rtol=0))


def parse_api_probabilities(raw, n_agents, purpose='distillation'):
    """Allow visible reasoning, but require a complete final all-agent JSON."""
    if purpose not in ('pure', 'distillation'):
        raise ValueError(purpose)
    if not isinstance(raw, str):
        raise ProbabilityFormatError('No final response text')
    answer = raw.strip()
    decoder = json.JSONDecoder()
    obj, span, offset = None, (-1,-1), 0
    while offset < len(answer):
        start = answer.find('{', offset)
        if start < 0: break
        try:
            candidate,end = decoder.raw_decode(answer,start)
        except ValueError:
            offset = start+1
            continue
        if isinstance(candidate,dict) and 'agents' in candidate:
            obj,span = candidate,(start,end)
        offset = end
    markers = list(re.finditer(r'"agents"\s*:\s*\[', answer))
    if markers and not (span[0] <= markers[-1].start() < span[1]):
        raise ProbabilityFormatError('Final agents JSON is incomplete')
    if obj is None or not isinstance(obj['agents'],list) or len(obj['agents']) != n_agents:
        raise ProbabilityFormatError('Missing all-agent JSON')
    rows = obj['agents']
    ids = [row.get('id') for row in rows if isinstance(row,dict)]
    if len(ids)!=n_agents or any(type(i) is not int for i in ids) or sorted(ids)!=list(range(n_agents)):
        raise ProbabilityFormatError('Missing, duplicate or invalid agent IDs')
    probabilities,repaired = [],False
    for row in sorted(rows,key=lambda row:row['id']):
        values = row.get('probs')
        if not isinstance(values,list) or len(values)!=6 or any(type(v) not in (float,int) for v in values):
            raise ProbabilityFormatError('Each agent requires exactly six numeric action entries')
        p = np.asarray(values,dtype=np.float64)
        if not np.isfinite(p).all():
            raise ProbabilityFormatError('Expected finite probabilities')
        if purpose=='pure':
            if (p<0).any() or (p>1).any() or not np.isclose(p.sum(),1,atol=1e-6,rtol=0):
                raise ProbabilityFormatError('Pure VLM probabilities must be within [0,1] and sum to one')
        else:
            p,fixed = clip_and_normalize(p)
            repaired = repaired or fixed
        probabilities.append(p)
    return np.asarray(probabilities,dtype=np.float32),repaired


def none_probabilities(n_agents):
    p = np.zeros((n_agents,6),dtype=np.float32)
    p[:,0] = 1.
    return p
