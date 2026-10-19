#!/usr/bin/env bash
set -euo pipefail
python -m pip install --target VLM4Coop/runtime/teacher_site --index-url https://download.pytorch.org/whl/cu118 torch==2.6.0 torchvision==0.21.0
# --target otherwise resolves a second, latest CUDA13 torch through accelerate.
# Install model libraries without changing the explicitly selected CUDA11 stack.
python -m pip install --no-deps --target VLM4Coop/runtime/teacher_site transformers==5.3.0 accelerate==1.12.0
python -m pip install --target VLM4Coop/runtime/teacher_site 'huggingface-hub>=1.3,<2' 'tokenizers>=0.22,<=0.23' 'safetensors>=0.4.3' regex requests tqdm packaging psutil PyYAML
bash VLM4Coop/scripts/finalize_teacher.sh
