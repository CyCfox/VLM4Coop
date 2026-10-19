#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# User explicitly authorized GPU 0 on 2026-10-17; do not expand to GPU 1.
exec docker run --rm --gpus device=0 --name "vlm4coop_gpu_$(date +%s)_cyc" \
  -u "$(id -u):$(id -g)" --cpus 4 --memory 40g --shm-size 1g \
  -v "$project_dir:/workspace/my_project:ro" \
  -v "$project_dir/VLM4Coop:/workspace/my_project/VLM4Coop:rw" \
  -v /media/data/chenyuchen/models/Qwen3.5-4B:/workspace/models/Qwen3.5-4B:ro \
  -w /workspace/my_project \
  -e HOME=/workspace/my_project/VLM4Coop/runtime/home \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e PYTHONPATH=/workspace/my_project/VLM4Coop/runtime/teacher_site:/workspace/my_project/VLM4Coop/runtime/site:/workspace/my_project/VLM4Coop/src:/workspace/my_project/epymarl/src \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/my_project/VLM4Coop/cache/torchinductor \
  -e TRITON_CACHE_DIR=/workspace/my_project/VLM4Coop/cache/triton \
  -e HF_HOME=/workspace/my_project/VLM4Coop/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel "$@"
