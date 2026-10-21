"""Fixed-scenario student evaluation, isolated from training RNG streams."""
import torch
from vlm4coop.utils.json_logging import append_json
from vlm4coop.utils.rng import evaluation_rng
from vlm4coop.evaluation.metrics import summarize

def evaluate(runner, config, output_dir, scheduled, actual, seed):
    start = config["fixed_eval_seeds"]["start"]
    count = config["eval_episodes_every_checkpoint"]
    if count % runner.batch_size:
        raise ValueError("Evaluation must contain full parallel batches")
    records = []
    # Same scenarios AND action RNG at every checkpoint. Restore training RNG.
    with evaluation_rng(start):
        for base in range(0, count, runner.batch_size):
            runner.next_episode_seeds = list(range(start + base, start + base + runner.batch_size))
            with torch.no_grad():
                runner.run(test_mode=True)
            for index, episode in enumerate(runner.last_episode_records):
                record = dict(episode, eval_seed=start + base + index, train_seed=seed,
                              scheduled_eval_step=scheduled, actual_train_env_steps=actual)
                append_json(output_dir / "eval_episodes.jsonl", record)
                records.append(record)
    summary = summarize(records)
    summary.update(scheduled_eval_step=scheduled, actual_train_env_steps=actual, train_seed=seed)
    append_json(output_dir / "evaluations.jsonl", summary)
    return summary

