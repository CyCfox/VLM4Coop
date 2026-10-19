#!/usr/bin/env bash
# The container owns training and postprocessing; API key arrives on stdin only.
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
run_dir=${1:?Pass a new VLM4Coop/results/hard_8x8_fixed500_1000k_api_* directory}
if [[ "$run_dir" != VLM4Coop/results/hard_8x8_fixed500_1000k_api_* || "$run_dir" == *..* || -e "$run_dir" ]]; then
  echo 'Expected a fresh project Hard results directory' >&2
  exit 2
fi
exec VLM4Coop/scripts/docker_api.sh bash VLM4Coop/scripts/hard_train_and_finish.sh "$run_dir"
