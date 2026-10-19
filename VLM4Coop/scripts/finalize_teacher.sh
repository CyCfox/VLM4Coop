#!/usr/bin/env bash
set -euo pipefail
# --target does not replace existing package dirs without --upgrade.
# Pin shared numerical/rendering dependencies after both installation stages.
python -m pip install --upgrade --no-deps --target VLM4Coop/runtime/teacher_site numpy==1.26.4 Pillow==11.3.0
python -m pip freeze --path VLM4Coop/runtime/teacher_site > VLM4Coop/reports/requirements-teacher-resolved.txt
