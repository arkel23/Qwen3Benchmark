#!/bin/bash
# World-model runs for the models that led the screening: each serves one model for all its runs.
# Two tiers (blind and briefed) and one repeat: at ~16 tok/s in agentic turns a run needs up to an hour.
# Usage: bash queue_worldmodel.sh M1 M2
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
E=/edahome/pcslab/pcs05/edwin
L=$E/model_bench/logs
CTX=131072
CAP=3600

while pgrep -f "serve_bench.py|run_model.sh|run_chat.py" > /dev/null; do sleep 60; done

for MODEL in "$@"; do
  python3 $HERE/serve.py stop
  python3 $HERE/serve.py start $MODEL --ctx $CTX --log $L/wm_server_$MODEL.log || { echo "$MODEL did not load"; continue; }
  for TASK in W1_render W2_classes; do
    for TIER in T1 T3; do
      for REPEAT in 0; do
        bash $HERE/run_worldmodel.sh $MODEL $TASK $TIER $REPEAT $CAP >> $L/wm_$MODEL.log 2>&1
      done
    done
  done
  python3 $HERE/serve.py stop
done
echo QUEUE_DONE_WM
