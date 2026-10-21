"""JSON serialization and scalar logging; no environment or training imports."""
from pathlib import Path
import json
import math

def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None  # JSON null denotes a missing completion statistic (NaN in CSV).
    return value


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(_json_safe(value), allow_nan=False) + "\n")


class JsonLogger:
    def __init__(self, path):
        self.path = path

    def log_stat(self, key, value, t):
        append_json(self.path, {"step": int(t), "metric": key, "value": float(value)})

