#!/usr/bin/env bash
# Stage-1a E1 gate runs, one after another on one GPU (ed520). Usage: scripts/run_stage1a_gates.sh <python> <runs_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=$1
RUNS=$2

for run in "combination full" "combination actions_only" "combination first_frame" "random full"; do
    set -- $run
    out="$RUNS/e1_$1_$2"
    mkdir -p "$out"
    echo "$(date '+%F %T') start $out"
    "$PYTHON" scripts/train_e1.py --split "$1" --input "$2" --out "$out" > "$out/train.log" 2>&1
    echo "$(date '+%F %T') done $out"
done
