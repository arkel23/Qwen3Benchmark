#!/bin/bash
# run_qwen.sh -- start the Qwen3.8-27B llama-server on camel and open Qwen Code in the current directory.
# Usage: ~/edwin/run_qwen.sh [qwen args]; the server stops when Qwen Code exits unless it was already running.
set -euo pipefail

E=/edahome/pcslab/pcs05/edwin
PORT=8080
LOG=$E/logs/llama_server.log
GGUF=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/27af057ecb382ddfea5d12837360a8980560e3ed/Qwen3.8-27B-UD-Q4_K_XL.gguf
export PATH=$E/tools/node-v22.23.2-linux-x64-glibc-217/bin:$PATH

if curl -sf http://127.0.0.1:$PORT/health >/dev/null; then
  echo "llama-server already running on :$PORT, reusing it"
else
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$FREE" -lt 31500 ]; then
    echo "GPU has ${FREE} MiB free; the 27B at ctx 229376 needs ~31100 MiB. In use:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    exit 1
  fi
  mkdir -p "$(dirname "$LOG")"
  $E/projects/llama.cpp/build/bin/llama-server -m "$GGUF" --host 127.0.0.1 --port $PORT \
    --ctx-size 229376 -np 1 -ngl 999 --jinja --reasoning-effort low -n 16384 \
    > "$LOG" 2>&1 < /dev/null &
  SERVER=$!
  trap 'echo "stopping llama-server"; kill $SERVER 2>/dev/null' EXIT
  echo -n "loading Qwen3.8-27B (log: $LOG)"
  until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
    if ! kill -0 $SERVER 2>/dev/null; then
      echo; echo "llama-server exited during load:" >&2; tail -5 "$LOG" >&2; exit 1
    fi
    echo -n "."; sleep 3
  done
  echo " ready"
fi

qwen "$@"
