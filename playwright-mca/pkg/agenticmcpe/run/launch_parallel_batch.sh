#!/bin/zsh
# Launch the 35-task benchmark as 9 parallel category groups, detached.
# Each group gets its own summary file and console log under runs/.
cd "$(dirname "$0")/../../.."   # repo root
PY=github-mcp-server-stable/pkg/venv/bin/python
RUNS=pkg/agenticmcpe/runs
mkdir -p "$RUNS"

launch() {
  local name="$1"; shift
  nohup $PY pkg/agenticmcpe/run/run_batch.py --summary=batch_${name}.json "$@" \
    > "$RUNS/console_${name}.log" 2>&1 &
  echo "launched $name pid=$!"
}

launch booking  playwright_booking
launch map_a    google_map_task_0001 google_map_task_0002 google_map_task_0003 google_map_task_0004
launch map_b    google_map_task_0005 google_map_task_0006 google_map_task_0007
launch hf_a     huggingface_task_0001 huggingface_task_0002 huggingface_task_0003 huggingface_task_0004
launch hf_b     huggingface_task_0005 huggingface_task_0006 huggingface_task_0007 huggingface_task_0008
launch paper_a  paper_task_0001 paper_task_0002 paper_task_0003 paper_task_0004
launch paper_b  paper_task_0005 paper_task_0006 paper_task_0007 paper_task_0008
launch sports_a sports_task_0001 sports_task_0002 sports_task_0003 sports_task_0004
launch sports_b sports_task_0005 sports_task_0006 sports_task_0007 sports_task_0008
