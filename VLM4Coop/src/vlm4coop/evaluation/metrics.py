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
    is_mpe=episodes[0].get('environment')=='mpe_simple_spread'
    fields = {
        "success_rate": [int(e["success"]) for e in episodes],
        "return": [e["team_return"] for e in episodes],
        "success_episodes_length": [] if is_mpe else [e["length"] for e in episodes if e["success"]],
        "episodes_length": [e["length"] for e in episodes],
    }
    if is_mpe:
        for metric in ('final_covered_landmarks','final_coverage_fraction','final_mean_min_distance',
                       'final_sum_min_distance','final_collision_pairs','collision_pairs_total'):
            fields[metric]=[e[metric] for e in episodes]
    for key, values in fields.items():
        result[key + "_mean"], result[key + "_std"] = _summary(values)
    return result


def checkpoint_rank(summary):
    if 'final_mean_min_distance_mean' in summary:
        return (summary['return_mean'],summary['success_rate_mean'],-summary['final_mean_min_distance_mean'])
    length = summary["success_episodes_length_mean"]
    return (summary["success_rate_mean"], summary["return_mean"],
            -length if math.isfinite(length) else -math.inf)
