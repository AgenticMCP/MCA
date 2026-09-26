#!/usr/bin/env bash
# Run the rectification pool across N parallel arms, then merge their KBs.
#
# Each arm gets its own runs dir, KB shard and progress file: run ids are
# timestamps, so two arms starting a task in the same second would otherwise
# share a work_dir and overwrite each other's artifacts. Arms are stride
# shards, so each sees the same mix of task lengths and sites.
#
#   ./launch_parallel_rectify.sh [ARMS] [MAX_REPLANS] [TAG]
#
# Resumable: re-running with the same TAG skips tasks already attempted.
set -u

ARMS="${1:-5}"
MAX_REPLANS="${2:-15}"
TAG="${3:-scale}"
shift $(( $# < 3 ? $# : 3 )) 2>/dev/null || true
EXTRA=("$@")   # forwarded to every arm, e.g. --exclude-category flight-price-lookup

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${RECTIFY_PYTHON:-python3}"
OUT="$HERE/runs_$TAG"
mkdir -p "$OUT"

echo "[launch] $ARMS arm(s), max_replans=$MAX_REPLANS, tag=$TAG -> $OUT"
cd "$REPO" || exit 1

pids=()
for ((i = 0; i < ARMS; i++)); do
  AGENTICMCPE_RUNS_DIR="$OUT/runs_arm$i" \
  PYTHONUNBUFFERED=1 \
  "$PY" -m pkg.taskgen_rectify rectify \
      --shard "$i/$ARMS" \
      --max-replans "$MAX_REPLANS" \
      --task-timeout 1800 \
      --kb "$OUT/kb_arm$i.json" \
      --progress "$OUT/progress_arm$i.jsonl" \
      --resume \
      ${EXTRA[@]+"${EXTRA[@]}"} \
      > "$OUT/arm$i.log" 2>&1 &
  pid=$!                      # bash 3.2 (macOS) has no ${arr[-1]}
  pids+=("$pid")
  echo "[launch] arm $i pid $pid -> $OUT/arm$i.log"
  sleep 3   # stagger browser startup
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done
echo "[launch] all arms finished ($fail arm(s) accepted nothing)"

shards=()
for ((i = 0; i < ARMS; i++)); do
  [[ -f "$OUT/kb_arm$i.json" ]] && shards+=("$OUT/kb_arm$i.json")
done
if ((${#shards[@]})); then
  # --runs-dirs turns on the coverage gate during the merge: shards keep every
  # entry an arm ever accepted, including ones written before the gate, so a
  # plain merge re-imports them on each run.
  "$PY" -m pkg.taskgen_rectify merge "${shards[@]}" \
      --into "$HERE/knowledge_base_rectified.json" \
      --runs-dirs "$OUT" "$HERE/../agenticmcpe/runs_rectify_test"
fi
"$PY" -m pkg.taskgen_rectify stats
