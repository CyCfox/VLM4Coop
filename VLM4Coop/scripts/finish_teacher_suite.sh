#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
container_name=${1:?exact project API container required}
suite_dir=${2:?suite output directory required}
[[ "$container_name" == vlm4coop_api_*_cyc ]]
[[ "$suite_dir" == VLM4Coop/results/go_formal_* && "$suite_dir" != *..* ]]
run_name=$(basename "$suite_dir")
docker wait "$container_name" > "VLM4Coop/reports/${run_name}_exit_code.txt"
if [[ $(< "VLM4Coop/reports/${run_name}_exit_code.txt") != 0 ]]; then
  {
    printf '\n## %s Go队列观察器记录失败\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf '%s\n' "- $suite_dir 容器非零退出，自动收尾未进入结果分析；应读取该目录status.json与原始日志后修复。"
    cat "$suite_dir/status.json"
    printf '\n'
  } >> CODEX_CHECKPOINT.md
  VLM4Coop/scripts/docker_cpu.sh python -c '
import json, sys
from pathlib import Path
p=Path("VLM4Coop/reports/PIPELINE_STATE.json")
s=json.loads(p.read_text())
s.update(formal_experiments="teacher_suite_failed", formal_container=None,
         failure_status=str(Path(sys.argv[1]) / "status.json"))
p.write_text(json.dumps(s, indent=2)+"\n")
' "$suite_dir"
  exit 1
fi
for key in soft_3 hard_3 soft_4 soft_5 pure_vlm_3; do
  VLM4Coop/scripts/docker_cpu.sh python -m vlm4coop.cli.validation.validate_outputs "$suite_dir/$key"
done
VLM4Coop/scripts/docker_analysis.sh python -m vlm4coop.evaluation.analysis \
  --runs "$suite_dir/runs.json" --output "${suite_dir}_analysis"
{
  printf '\n## %s Go正式队列自动收尾完成\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf '%s\n' "- 队列：$suite_dir；全部训练/Pure阶段正常退出后才统一核验与制图。"
  printf '%s\n' "- 六组比较数据、表格与曲线在${suite_dir}_analysis；需主agent读取原始证据并审计后写论文，不能仅凭退出码宣称方法优越。"
} >> CODEX_CHECKPOINT.md
echo TEACHER_SUITE_POSTPROCESS_COMPLETE
