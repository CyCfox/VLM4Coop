#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/docker_cpu.sh" env \
  PYTHONPATH=/workspace/my_project/VLM4Coop/runtime/analysis_site:/workspace/my_project/VLM4Coop/runtime/site:/workspace/my_project/VLM4Coop/src:/workspace/my_project/epymarl/src \
  "$@"
