"""Record an explicit user stop after the project's container has exited."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

project = Path('VLM4Coop')
suite = project / 'results/go_formal_8x8_1000k_recovery_20261017_01'
stage = suite / 'soft_3'
now = datetime.now(timezone(timedelta(hours=8))).isoformat()
reason = 'user_reported_opencode_go_5_hour_limit_stop_requested'
state = json.loads((suite / 'status.json').read_text())
state.update(status='stopped_by_user', stopped_at=now, stop_reason=reason, auto_restart=False)
(suite / 'status.json').write_text(json.dumps(state, indent=2))
training = json.loads((stage / 'status.json').read_text())
sampled = 0
with (stage / 'train_episodes.jsonl').open() as stream:
    for line in stream:
        if line.strip():
            sampled = max(sampled, json.loads(line)['actual_train_env_steps'])
training.update(status='stopped_by_user', stopped_at=now, stop_reason=reason,
                sampled_env_steps=sampled, auto_restart=False)
(stage / 'status.json').write_text(json.dumps(training, indent=2))
checkpoints = sorted(int(p.name) for p in (stage / 'checkpoints').iterdir() if p.is_dir())
record = dict(status='stopped_by_user', stopped_at=now, reason=reason,
    suite=str(suite), last_reported_optimized_env_steps=training['train_env_steps'],
    sampled_env_steps=sampled, retained_checkpoint_steps=checkpoints,
    raw_logs_cache_and_checkpoints_preserved=True, auto_restart=False,
    exact_resume_supported=False,
    note='Existing checkpoints do not preserve worker environment RNG; no new checkpoint was fabricated at stop.')
(suite / 'interruption.json').write_text(json.dumps(record, indent=2))
ledger = project / 'reports/PIPELINE_STATE.json'
metadata = json.loads(ledger.read_text())
metadata.update(formal_experiments='stopped_by_user', formal_container=None,
                user_pause_reason=reason, auto_restart=False,
                interruption_record=str(suite / 'interruption.json'))
ledger.write_text(json.dumps(metadata, indent=2))
print(json.dumps(record, indent=2))
