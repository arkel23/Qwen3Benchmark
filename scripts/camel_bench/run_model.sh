#!/bin/bash
# Screen one model on camel: start its server, run the chat tasks, stop the server, grade.
# Usage: bash run_model.sh M0 [ctx] [n_cpu_moe]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/edahome/pcslab/pcs05/edwin/model_bench
MODEL=$1
CTX=${2:-131072}
NCPUMOE=${3:-0}
mkdir -p $OUT/raw $OUT/graded $OUT/logs

nvidia-smi --query-gpu=memory.used --format=csv,noheader
python3 $HERE/serve.py start $MODEL --ctx $CTX --n_cpu_moe $NCPUMOE --log $OUT/logs/server_$MODEL.log
trap 'python3 $HERE/serve.py stop' EXIT
nvidia-smi --query-gpu=memory.used --format=csv,noheader
python3 $HERE/run_chat.py --model_id $MODEL --out $OUT/raw/$MODEL.jsonl
python3 $HERE/serve.py stop
trap - EXIT

source /edahome/pcslab/pcs05/miniconda3/etc/profile.d/conda.sh && conda activate camel_bench
python $HERE/grade.py --raw $OUT/raw/$MODEL.jsonl --out $OUT/graded/$MODEL.csv
echo "SCREEN_DONE $MODEL"
