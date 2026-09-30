#!/bin/bash
# Flash-Next: keep the fastest CPU expert offload that fits the card without swapping, then screen it once.
# Decode speed is not monotonic in the offload level: fewer CPU layers means more PCIe traffic per token.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
E=/edahome/pcslab/pcs05/edwin
L=$E/model_bench/logs
CTX=131072

while pgrep -f queue_screen.sh > /dev/null; do sleep 60; done

BEST=""
for N in 48 44 40 36 32; do
  echo "== n_cpu_moe $N"
  if ! python3 $HERE/serve.py start M4 --ctx $CTX --n_cpu_moe $N --log $L/m4_tune_$N.log; then
    echo "  did not load"; python3 $HERE/serve.py stop; continue
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  free -g | awk '/^Mem|^Swap/ {print "  " $1 " used " $3 " GiB"}'
  TPS=$(curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from 1 to 40."}],"max_tokens":128}' \
    | python3 -c "import json,sys; print(round(json.load(sys.stdin)['timings']['predicted_per_second'], 2))")
  SWAP=$(free -g | awk '/^Swap/ {print $3}')
  python3 $HERE/serve.py stop
  echo "  decode $TPS tok/s, swap ${SWAP} GiB"
  if [ "${SWAP:-0}" -le 1 ]; then
    BESTTPS=${BEST#* }
    if [ -z "$BEST" ] || [ "${TPS%%.*}" -gt "${BESTTPS%%.*}" ]; then BEST="$N $TPS"; fi
  fi
done

echo "chosen: ${BEST:-none}"
[ -z "$BEST" ] && { echo "M4 does not fit"; exit 1; }
N=${BEST%% *}
bash $HERE/run_model.sh M4 $CTX $N > $L/screen_M4.log 2>&1 || true
bash -c "source /edahome/pcslab/pcs05/miniconda3/etc/profile.d/conda.sh; python3 $HERE/serve_bench.py M4 --n_cpu_moe $N" > $L/speed_M4.log 2>&1 || true
echo QUEUE_DONE_M4
