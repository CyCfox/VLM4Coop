#!/usr/bin/env bash
# Container owns the complete lifecycle; never print the credential.
set -uo pipefail
run_dir=${1:?new MPE Soft run directory required}
if [[ "$run_dir" != VLM4Coop/results/soft_mpe_spread_fixed500_1000k_* || "$run_dir" == *..* || -e "$run_dir" ]]; then
  echo 'Expected a fresh project MPE Soft directory' >&2
  exit 2
fi
run_name=$(basename "$run_dir")
exec > "VLM4Coop/reports/${run_name}.log" 2>&1
python -u -m vlm4coop.training.loop --method soft --agents 3 --seed 0 --profile formal \
  --config VLM4Coop/configs/mpe_soft1000.yaml --output "$run_dir"
code=$?
unset OPENCODE_API_KEY
printf '%s\n' "$code" > "VLM4Coop/reports/${run_name}_exit_code.txt"
if [[ "$code" -ne 0 ]]; then exit "$code"; fi
python -m vlm4coop.cli.validation.validate_outputs "$run_dir" \
  > "VLM4Coop/reports/${run_name}_validation.log" 2>&1 || exit $?
PYTHONPATH="/workspace/my_project/VLM4Coop/runtime/analysis_site:$PYTHONPATH" \
python -m vlm4coop.cli.analysis.export_training_run "$run_dir" --output "${run_dir}_analysis" \
  > "VLM4Coop/reports/${run_name}_analysis.log" 2>&1 || exit $?
python -m vlm4coop.cli.diagnostics.record_mpe_soft completion "$run_dir" || exit $?
echo 'MPE_SOFT_1000K_COMPLETE_VALIDATED_EXPORTED'
