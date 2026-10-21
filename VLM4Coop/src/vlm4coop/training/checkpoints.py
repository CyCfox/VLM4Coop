"""Native model snapshots and retention, without training orchestration."""
import json
import shutil
import torch
from vlm4coop.evaluation.metrics import checkpoint_rank
from vlm4coop.utils.json_logging import _json_safe

def _snapshot(learner, output, actual, summary, best, retain, manifest):
    checkpoint_dir = output / "checkpoints" / str(actual)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
        learner.save_models(str(checkpoint_dir))
        # Native save_models lacks target network/running moments; preserve them.
        extra = {"target_critic": learner.target_critic.state_dict(),
                 "old_mac": learner.old_mac.agent.state_dict(),
                 "critic_training_steps": learner.critic_training_steps}
        for name in ("rew_ms", "ret_ms"):
            if hasattr(learner, name):
                obj = getattr(learner, name)
                extra[name] = {k: getattr(obj, k) for k in ("mean", "var", "count")}
        torch.save(extra, checkpoint_dir / "learner_extra.th")
    manifest.append(actual)
    current_rank = checkpoint_rank(summary)
    if best is None or current_rank > best:
        best = current_rank
        # The best pointer keeps its directory from retention cleanup.
        (output / "best_checkpoint.json").write_text(json.dumps(_json_safe({"train_env_steps": actual,
                                                                 "rank": list(best)}), allow_nan=False))
    best_step = json.loads((output / "best_checkpoint.json").read_text())["train_env_steps"]
    keep = set(manifest[-retain:]) | {best_step}
    for path in (output / "checkpoints").iterdir():
        if path.is_dir() and int(path.name) not in keep:
            shutil.rmtree(path)  # only this run's explicitly-owned checkpoints
    return best

