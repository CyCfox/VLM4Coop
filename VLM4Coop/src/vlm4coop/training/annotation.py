"""Freeze one teacher target per sampled state before PPO epochs."""
from dataclasses import replace
import json
import numpy as np
import torch
from vlm4coop.utils.rgb_frames import frame_image

def annotate(runner, batch, teacher):
    """Freeze pre-action targets once; allow the configured HTTP concurrency."""
    try:
        items = list(runner.frames.items())
        chunk_size = max(32, getattr(teacher, 'config', {}).get('concurrency', 1))
        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            if any(t >= batch.max_seq_length - 1 for (worker, t), rgb in chunk):
                raise ValueError("Captured terminal/padding frame")
            images = [frame_image(rgb) for _, rgb in chunk]
            if hasattr(teacher, 'predict_many'):
                results = teacher.predict_many(images, runner.args.n_agents)
            else:
                results = [teacher.predict(image, runner.args.n_agents) for image in images]
            if len(results) != len(chunk):
                raise ValueError("Teacher returned a different number of state labels")
            resolved=[]
            for ((worker, t), _), result in zip(chunk, results):
                if result.source=='format_failure':
                    # Storage placeholder only, not a teacher distribution.
                    # PPO retains its native rollout mask; only KD is disabled.
                    result=replace(result,probs=np.zeros((runner.args.n_agents,runner.args.n_actions),dtype=np.float32),source='kd_masked')
                resolved.append(result)
                batch.update({"teacher_probs": torch.from_numpy(result.probs)[None, None],
                              "teacher_valid": torch.full((1,1,1),float(result.source!='kd_masked'))},
                             bs=[worker], ts=t, mark_filled=False)
            if hasattr(teacher, 'audit_path'):
                with teacher.audit_path.with_name('teacher_targets.jsonl').open('a') as stream:
                    for ((worker, t), _), image, result in zip(chunk, images, resolved):
                        record = dict(actual_train_env_steps=runner.t_env, worker=worker, t=t,
                            key=teacher.cache.key(image, runner.args.n_agents) if teacher.cache else None,
                            probabilities=result.probs.tolist() if result.source!='kd_masked' else None,
                            teacher_valid=int(result.source!='kd_masked'),cache_hit=result.cache_hit,
                            repaired=result.repaired,source=result.source,
                            format_attempts=result.format_attempts,decision_id=result.decision_id)
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
    finally:
        runner.frames.clear()

