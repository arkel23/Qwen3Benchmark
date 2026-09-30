#!/bin/bash
# Screen the remaining models one at a time after M0, each at 131072 context; Flash-Next gets one repeat.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
L=/edahome/pcslab/pcs05/edwin/model_bench/logs

while pgrep -f "run_chat.py --model_id M0" > /dev/null; do sleep 60; done
bash -c "source /edahome/pcslab/pcs05/miniconda3/etc/profile.d/conda.sh; python3 $HERE/serve_bench.py M0" > $L/speed_M0.log 2>&1
for MODEL in M1 M2 M3; do
  bash $HERE/run_model.sh $MODEL 131072 > $L/screen_$MODEL.log 2>&1
  bash -c "source /edahome/pcslab/pcs05/miniconda3/etc/profile.d/conda.sh; python3 $HERE/serve_bench.py $MODEL" > $L/speed_$MODEL.log 2>&1
done
echo QUEUE_DONE_SMALL
