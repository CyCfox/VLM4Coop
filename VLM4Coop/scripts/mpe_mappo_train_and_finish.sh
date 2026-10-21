#!/usr/bin/env bash
# Container-owned train -> validate -> export, without an intermediate observer.
set -uo pipefail
run_dir=${1:?new MPE MAPPO run directory required}
config_file=${2:-VLM4Coop/configs/mpe_spread.yaml}
if [[ "$run_dir" != VLM4Coop/results/mappo_mpe_spread_fixed500_1000k_* || "$run_dir" == *..* || -e "$run_dir" ]]; then
  echo 'Expected a fresh MPE MAPPO directory' >&2
  exit 2
fi
run_name=$(basename "$run_dir")
exec > "VLM4Coop/reports/${run_name}.log" 2>&1
python -u -m vlm4coop.training.loop --method mappo --agents 3 --seed 0 --profile formal \
  --config "$config_file" --output "$run_dir"
code=$?
printf '%s\n' "$code" > "VLM4Coop/reports/${run_name}_exit_code.txt"
if [[ "$code" -ne 0 ]]; then exit "$code"; fi
python -m vlm4coop.cli.validation.validate_outputs "$run_dir" \
  > "VLM4Coop/reports/${run_name}_validation.log" 2>&1 || exit $?
PYTHONPATH="/workspace/my_project/VLM4Coop/runtime/analysis_site:$PYTHONPATH" \
python -m vlm4coop.cli.analysis.export_training_run "$run_dir" --output "${run_dir}_analysis" \
  > "VLM4Coop/reports/${run_name}_analysis.log" 2>&1 || exit $?
python -c 'import sys; from vlm4coop.cli.diagnostics.record_mpe_launch import record_completion; record_completion(sys.argv[1])' "$run_dir" || exit $?
echo 'MPE_MAPPO_1000K_COMPLETE_VALIDATED_EXPORTED'
