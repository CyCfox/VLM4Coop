#!/usr/bin/env bash
# Sole authorized MPE Soft run; credential arrives through stdin only.
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
run_dir=${1:?Pass a new VLM4Coop/results/soft_mpe_spread_fixed500_1000k_* directory}
if [[ "$run_dir" != VLM4Coop/results/soft_mpe_spread_fixed500_1000k_* || "$run_dir" == *..* || -e "$run_dir" ]]; then
  echo 'Expected a fresh project MPE Soft directory' >&2
  exit 2
fi
exec VLM4Coop/scripts/docker_mpe_api.sh bash VLM4Coop/scripts/mpe_soft_train_and_finish.sh "$run_dir"
