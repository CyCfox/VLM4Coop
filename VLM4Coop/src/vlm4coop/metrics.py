"""Episode statistics: unsuccessful lengths never enter completion time."""
import math
import numpy as np


def _summary(values):
    values = np.asarray(values, dtype=float)
    return (float(values.mean()) if len(values) else math.nan,
            float(values.std(ddof=1)) if len(values) > 1 else math.nan)


def summarize(episodes):
    if not episodes:
        raise ValueError("Cannot summarize zero evaluation episodes")
    result = {"n_eval": len(episodes),
              "n_success_episodes": sum(bool(e["success"]) for e in episodes)}
    result["n_failed_episodes"] = result["n_eval"] - result["n_success_episodes"]
    fields = {
        "success_rate": [int(e["success"]) for e in episodes],
        "return": [e["team_return"] for e in episodes],
        "success_episodes_length": [e["length"] for e in episodes if e["success"]],
        "episodes_length": [e["length"] for e in episodes],
    }
    for key, values in fields.items():
        result[key + "_mean"], result[key + "_std"] = _summary(values)
    return result


def checkpoint_rank(summary):
    length = summary["success_episodes_length_mean"]
    return (summary["success_rate_mean"], summary["return_mean"],
            -length if math.isfinite(length) else -math.inf)
