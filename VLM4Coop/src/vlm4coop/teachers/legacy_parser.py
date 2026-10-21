"""Unchanged historical local-Qwen LBF parser."""
import json
import re
import numpy as np

def parse_probabilities(raw, n_agents):
    # Legacy local-Qwen/archive parser. Active API experiments use the strict,
    # role-specific parser in response_protocol.py.
    # Thinking can contain drafts; parse only the final answer after </think>.
    answer = raw.rsplit("</think>", 1)[-1].strip()
    decoder = json.JSONDecoder()
    obj = None
    obj_span = (-1, -1)
    offset = 0
    # Content may contain prose, fenced JSON and draft objects. Decode whole
    # objects and select the final all-agent object rather than a reasoning draft.
    while offset < len(answer):
        start = answer.find("{", offset)
        if start < 0:
            break
        try:
            candidate, end = decoder.raw_decode(answer, start)
        except ValueError:
            offset = start + 1
            continue
        if isinstance(candidate, dict) and "agents" in candidate:
            obj = candidate
            obj_span = (start, end)
        offset = end
    syntax_repaired = False
    markers = list(re.finditer(r'"agents"\s*:\s*\[', answer))
    if markers and not (obj_span[0] <= markers[-1].start() < obj_span[1]):
        # Observed API response: every agent object and six-value row was
        # complete, but the enclosing agents list lacked its closing bracket.
        # Recover only complete JSON row objects from the final agents list;
        # never infer an absent ID, action entry or probability value.
        tail = answer[markers[-1].end():]
        recovered, cursor = [], 0
        while cursor < len(tail):
            start = tail.find('{', cursor)
            if start < 0:
                break
            try:
                row, end = decoder.raw_decode(tail, start)
            except ValueError:
                raise ValueError('Incomplete agent probability object') from None
            if not isinstance(row, dict) or 'id' not in row or 'probs' not in row:
                raise ValueError('Cannot identify complete agent probability rows')
            recovered.append(row)
            cursor = end
        obj = {'agents': recovered}
        syntax_repaired = True
    if obj is None or not isinstance(obj["agents"], list) or len(obj["agents"]) != n_agents:
        raise ValueError("Missing all-agent JSON")
    rows = obj["agents"]
    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(ids) != n_agents or any(type(i) is not int for i in ids) or sorted(ids) != list(range(n_agents)):
        raise ValueError("Missing, duplicate or invalid agent IDs")
    probs, repaired = [], syntax_repaired
    for row in sorted(rows, key=lambda item: item["id"]):
        values = row.get("probs")
        if not isinstance(values, list) or not values:
            raise ValueError("Cannot identify action entries")
        numeric = np.asarray([float(v) if type(v) in (int, float) else np.nan for v in values])
        if not np.isfinite(numeric).any():
            raise ValueError("No identifiable finite maximum")
        maximum = int(np.nanargmax(numeric))
        confidence = numeric[maximum]
        if maximum >= 6:
            raise ValueError("Maximum has no legal action index; cannot preserve it")
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Maximum is not a legal probability; cannot preserve it")
        p = numeric.copy() if len(numeric) == 6 else np.empty(6, dtype=float)
        valid = len(numeric) == 6 and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        if not valid or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
            p[:] = (1 - confidence) / 5
            p[maximum] = confidence
            repaired = True
        p = p / p.sum()  # floating-point roundoff only
        probs.append(p)
    return np.asarray(probs, dtype=np.float32), repaired

