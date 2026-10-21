#!/usr/bin/env bash
# Run inside Docker. Persist logs and finish outputs without a host observer.
set -uo pipefail
run_dir=${1:?new Hard results directory required}
if [[ "$run_dir" != VLM4Coop/results/hard_8x8_fixed500_1000k_api_* || "$run_dir" == *..* || -e "$run_dir" ]]; then
  echo 'Expected a fresh project Hard results directory' >&2
  exit 2
fi
run_name=$(basename "$run_dir")
exec > "VLM4Coop/reports/${run_name}.log" 2>&1
python -u -m vlm4coop.training.loop --method hard --agents 3 --seed 0 --profile formal --output "$run_dir"
train_exit=$?
unset OPENCODE_API_KEY
printf '%s\n' "$train_exit" > "VLM4Coop/reports/${run_name}_exit_code.txt"
if [[ "$train_exit" -ne 0 ]]; then
  echo 'Hard training failed; preserving raw outputs without automatic restart' >&2
  exit "$train_exit"
fi
python -m vlm4coop.cli.validation.validate_outputs "$run_dir" \
  > "VLM4Coop/reports/${run_name}_validation.log" 2>&1 || exit $?
python -m vlm4coop.cli.analysis.export_training_run "$run_dir" --output "${run_dir}_analysis" \
  > "VLM4Coop/reports/${run_name}_analysis.log" 2>&1 || exit $?
echo 'FORMAL_HARD_1000K_COMPLETE_AND_EXPORTED'
