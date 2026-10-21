#!/usr/bin/env bash
# One explicitly authorized main experiment; credentials arrive only on stdin.
set -uo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
run_dir=${1:?Pass a new VLM4Coop/results/soft_8x8_fixed500_1000k_api_* directory}
if [[ "$run_dir" != VLM4Coop/results/soft_8x8_fixed500_1000k_api_* || "$run_dir" == *..* || -e "$run_dir" ]]; then
  echo 'Expected a fresh project Soft results directory' >&2
  exit 2
fi
run_name=$(basename "$run_dir")
VLM4Coop/scripts/docker_api.sh python -u -m vlm4coop.training.loop \
  --method soft --agents 3 --seed 0 --profile formal --output "$run_dir" \
  > "VLM4Coop/reports/${run_name}.log" 2>&1
train_exit=$?
printf '%s\n' "$train_exit" > "VLM4Coop/reports/${run_name}_exit_code.txt"
if [[ "$train_exit" -ne 0 ]]; then
  echo "Formal Soft training failed; preserve $run_dir and inspect its original log" >&2
  exit "$train_exit"
fi
VLM4Coop/scripts/docker_cpu.sh python -m vlm4coop.cli.validation.validate_outputs "$run_dir" \
  > "VLM4Coop/reports/${run_name}_validation.log" 2>&1 || exit $?
VLM4Coop/scripts/docker_cpu.sh python -m vlm4coop.cli.analysis.export_training_run \
  "$run_dir" --output "${run_dir}_analysis" \
  > "VLM4Coop/reports/${run_name}_analysis.log" 2>&1 || exit $?
echo 'FORMAL_SOFT_1000K_COMPLETE_AND_EXPORTED'
