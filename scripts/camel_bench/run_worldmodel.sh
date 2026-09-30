#!/bin/bash
# One world-model run: fresh fixture copy, headless Qwen Code writes the missing file, then grade it.
# Usage: bash run_worldmodel.sh <model_id> <W1_render|W2_classes> <T1|T2|T3> <repeat> [wall_cap_s]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
E=/edahome/pcslab/pcs05/edwin
OUT=$E/model_bench/wm
MODEL=$1; TASK=$2; TIER=$3; REPEAT=$4; CAP=${5:-2400}
RUN=$OUT/${MODEL}_${TASK}_${TIER}_r${REPEAT}
TARGET=$([ "$TASK" = W1_render ] && echo pong/render.py || echo pong/classes.py)

rm -rf "$RUN"; mkdir -p "$RUN"
cp -r $HERE/tasks/worldmodel/snapshot "$RUN/repo"
rm -rf "$RUN/repo/pong/__pycache__"
rm "$RUN/repo/$TARGET"
[ "$TASK" = W1_render ] && rm "$RUN/repo/pong/torch_render.py"

cd "$RUN/repo"
START=$(date +%s)
timeout $CAP $E/qwen --approval-mode=yolo -o text \
  -p "$(cat $HERE/tasks/worldmodel/$TASK/$TIER.md)" < /dev/null > "$RUN/qwen.log" 2>&1 || echo "qwen exit $?" >> "$RUN/qwen.log"
echo "$(( $(date +%s) - START ))" > "$RUN/wall_s"

source /edahome/pcslab/pcs05/miniconda3/etc/profile.d/conda.sh && conda activate camel_bench
python $HERE/grade_worldmodel.py --run "$RUN" --model_id "$MODEL" --task "$TASK" --tier "$TIER" --repeat "$REPEAT" \
  --out $E/model_bench/worldmodel.csv
echo "WM_DONE $MODEL $TASK $TIER r$REPEAT"
