#!/usr/bin/env bash
# One authorized MPE MAPPO run; no API/GPU. All work and exports stay in Docker.
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
run_dir=${1:?Pass a fresh VLM4Coop/results/mappo_mpe_spread_fixed500_1000k_* directory}
config_file=${2:-VLM4Coop/configs/mpe_spread.yaml}
if [[ "$run_dir" != VLM4Coop/results/mappo_mpe_spread_fixed500_1000k_* || "$run_dir" == *..* || -e "$run_dir" ]]; then
  echo 'Expected a new project-owned MPE MAPPO results directory' >&2
  exit 2
fi
exec VLM4Coop/scripts/docker_mpe.sh bash VLM4Coop/scripts/mpe_mappo_train_and_finish.sh "$run_dir" "$config_file"
