#!/bin/bash
# Per model: four single-turn world-model tasks, then one agentic run as a tool-use probe.
# Usage: bash queue_worldmodel2.sh M1 M2
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
E=/edahome/pcslab/pcs05/edwin
L=$E/model_bench/logs

for MODEL in "$@"; do
  python3 $HERE/serve.py stop
  python3 $HERE/serve.py start $MODEL --ctx 131072 --log $L/wm_server_$MODEL.log || { echo "$MODEL did not load"; continue; }
  for TASK in W1_render W2_classes; do
    for TIER in T1 T3; do
      python3 $HERE/run_worldmodel_single.py --model_id $MODEL --task $TASK --tier $TIER >> $L/wm_single_$MODEL.log 2>&1
      echo "SINGLE_DONE $MODEL $TASK $TIER"
    done
  done
  bash $HERE/run_worldmodel.sh $MODEL W1_render T3 0 3600 >> $L/wm_$MODEL.log 2>&1
  python3 $HERE/serve.py stop
done
echo QUEUE_DONE_WM2
