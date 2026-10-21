"""Persist the completed prelaunch checks, without touching experiment logs."""
import json
from pathlib import Path
p=Path('VLM4Coop/reports/PIPELINE_STATE.json')
s=json.loads(p.read_text())
s.update(distillation_formal_training='hard_relaunch_validation_complete_waiting_fresh_review',
    current_tests='79_passed_63.12s_Docker',response_protocol='strict-schema-role-specific-v6')
s['thinking'].update(returned_reasoning='accept_if_JSON_valid_or_repairable',control_retries=0)
s['hard_main']['restart_pending']='results/hard_8x8_fixed500_1000k_api_20261019_02'
s['hard_main']['new_preflight']='results/hard_accept_reasoning_preflight_20261019_01'
p.write_text(json.dumps(s,indent=2))
