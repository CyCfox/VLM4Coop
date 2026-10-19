#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# stdin delivers the key to the container shell, never in docker args/config.
exec docker run --rm -i --name "vlm4coop_api_$(date +%s)_cyc" \
  -u "$(id -u):$(id -g)" --cpus 4 --memory 8g \
  -v "$project_dir:/workspace/my_project:ro" \
  -v "$project_dir/VLM4Coop:/workspace/my_project/VLM4Coop:rw" \
  -w /workspace/my_project \
  -e HOME=/workspace/my_project/VLM4Coop/runtime/home \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e PYTHONPATH=/workspace/my_project/VLM4Coop/runtime/site:/workspace/my_project/VLM4Coop/src:/workspace/my_project/epymarl/src \
  pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
  bash -c 'IFS= read -r OPENCODE_API_KEY; export OPENCODE_API_KEY; exec "$@"' bash "$@"
