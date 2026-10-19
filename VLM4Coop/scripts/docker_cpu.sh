#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec docker run --rm --name "vlm4coop_cpu_$(date +%s)_cyc" \
  -u "$(id -u):$(id -g)" --cpus 4 --memory 8g \
  -v "$project_dir:/workspace/my_project:ro" \
  -v "$project_dir/VLM4Coop:/workspace/my_project/VLM4Coop:rw" \
  -v /media/data/chenyuchen/models/Qwen3.5-4B:/workspace/models/Qwen3.5-4B:ro \
  -w /workspace/my_project \
  -e HOME=/workspace/my_project/VLM4Coop/runtime/home \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e PYTHONPATH=/workspace/my_project/VLM4Coop/runtime/site:/workspace/my_project/VLM4Coop/src:/workspace/my_project/epymarl/src \
  -e PIP_CACHE_DIR=/workspace/my_project/VLM4Coop/cache/pip \
  pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime "$@"
