#!/usr/bin/env bash
# Wait for this project's named training container; never inspect live metrics.
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
container_name=${1:?Pass the exact project MAPPO container name}
if [[ "$container_name" != vlm4coop_cpu_*_cyc ]]; then
  echo 'Refusing a container outside this project CPU naming pattern' >&2
  exit 2
fi
run_dir=${2:-VLM4Coop/results/mappo_8x8_formal_20261017_01}
if [[ "$run_dir" != VLM4Coop/results/mappo_8x8_formal_* ]] || [[ "$run_dir" == *..* ]]; then
  echo 'Refusing output outside this project MAPPO results pattern' >&2
  exit 2
fi
run_name=$(basename "$run_dir")
report_dir=VLM4Coop/reports
exit_file="$report_dir/${run_name}_exit_code.txt"
docker wait "$container_name" > "$exit_file"
if [[ $(< "$exit_file") != 0 ]]; then
  {
    printf '\n## %s MAPPO观察器记录失败\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf '%s\n' "- $run_dir 容器非零退出；读取status.json及控制台日志后处理，不自动重启或请求教师API。"
  } >> CODEX_CHECKPOINT.md
  echo "Training exited unsuccessfully; see $exit_file and the original log" >&2
  exit 1
fi
VLM4Coop/scripts/docker_cpu.sh python -m vlm4coop.cli.validation.validate_outputs "$run_dir"
VLM4Coop/scripts/docker_analysis.sh python -m vlm4coop.cli.analysis.summarize_mappo "$run_dir" \
  --output "${run_dir}_analysis"
VLM4Coop/scripts/docker_cpu.sh python -c '
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
config = json.loads((run / "config.json").read_text())
p = Path("VLM4Coop/reports/PIPELINE_STATE.json")
state = json.loads(p.read_text())
state.update(formal_experiments="mappo_complete_teacher_experiments_paused",
             result_analysis="single_mappo_baseline_complete_no_teacher_comparison",
             formal_container=None,
             formal_run=str(run.relative_to("VLM4Coop")),
             formal_train_steps=config["experiment_config"]["formal_train_steps"],
             baseline_analysis=str(run.relative_to("VLM4Coop")) + "_analysis")
p.write_text(json.dumps(state, indent=2) + "\n")
' "$run_dir"
{
  printf '\n## %s 正式MAPPO自动收尾完成\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf '%s\n' '- 等待本项目训练容器自然退出code0后才读取结果，没有轮询中间成绩。'
  printf '%s\n' '- validate_outputs与summarize_mappo均返回0；正式原始记录和统计重算通过。'
  printf '%s\n' "- 结果：$run_dir；统计与PDF/SVG/PNG曲线在同名_analysis目录。"
  printf '%s\n' '- 这里只完成MAPPO基线，尚未完成教师正式比较和论文。下一步人工/主agent核验REPORT.md、validation.json、最终曲线并向用户报告。'
} >> CODEX_CHECKPOINT.md
echo 'MAPPO_POSTPROCESS_COMPLETE'
