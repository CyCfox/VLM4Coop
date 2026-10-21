#!/usr/bin/env bash
# Isolated MPE layer; no GPU, API credential, model mount or port allocation.
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
stdin_args=()
if [[ "${1:-}" == --stdin ]]; then stdin_args=(-i); shift; fi
exec docker run --rm "${stdin_args[@]}" --sig-proxy=false --name "vlm4coop_mpe_$(date +%s)_cyc" \
  -u "$(id -u):$(id -g)" --cpus 4 --memory 8g \
  -v "$project_dir:/workspace/my_project:ro" \
  -v "$project_dir/VLM4Coop:/workspace/my_project/VLM4Coop:rw" \
  -w /workspace/my_project \
  -e HOME=/workspace/my_project/VLM4Coop/runtime/home \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e SDL_VIDEODRIVER=dummy -e SDL_AUDIODRIVER=dummy -e PYGAME_HIDE_SUPPORT_PROMPT=1 \
  -e PYTHONPATH=/workspace/my_project/VLM4Coop/runtime/mpe_site:/workspace/my_project/VLM4Coop/runtime/site:/workspace/my_project/VLM4Coop/src:/workspace/my_project/epymarl/src \
  pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime "$@"
