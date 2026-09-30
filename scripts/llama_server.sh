#!/bin/bash
# llama_server.sh -- start, stop or check the Qwen3.8-27B llama-server on camel; it keeps running after logout.
# Usage: ~/edwin/llama_server.sh [start [--model q5|q6] [--slots N] [--ctx N] [--kv f16|q8_0]|stop|status]
set -euo pipefail

E=/edahome/pcslab/pcs05/edwin
PORT=8080
LOG=$E/logs/llama_server.log
PIDFILE=$E/logs/llama_server.pid
SNAPSHOT=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502

ACTION=${1:-start}
[ $# -gt 0 ] && shift
MODEL=q5
SLOTS=1
CTX=""
KV=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --slots) SLOTS=$2; shift 2 ;;
    --ctx) CTX=$2; shift 2 ;;
    --kv) KV=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

# Defaults are the largest context that fits 32 GB for both 1 and 4 slots (docs/local_models/qwen_v100_USAGE.md).
case "$MODEL" in
  q5) GGUF=$SNAPSHOT/Qwen3.8-27B-UD-Q5_K_XL.gguf; CTX=${CTX:-196608}; KV=${KV:-f16} ;;
  q6) GGUF=$SNAPSHOT/Qwen3.8-27B-UD-Q6_K_XL.gguf; CTX=${CTX:-196608}; KV=${KV:-q8_0} ;;
  *) echo "unknown model $MODEL (use q5 or q6)" >&2; exit 2 ;;
esac

healthy() { curl -sf http://127.0.0.1:$PORT/health >/dev/null; }
our_pid() { [ -f "$PIDFILE" ] && ps -p "$(cat "$PIDFILE")" -o cmd= 2>/dev/null | grep -q '[l]lama-server' && cat "$PIDFILE"; }

case "$ACTION" in
start)
  if healthy; then echo "llama-server already running on :$PORT"; exit 0; fi
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$FREE" -lt 31500 ]; then
    echo "GPU has ${FREE} MiB free; the 27B needs the whole card. In use:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    exit 1
  fi
  mkdir -p "$(dirname "$LOG")"
  nohup setsid $E/projects/llama.cpp/build/bin/llama-server -m "$GGUF" --host 127.0.0.1 --port $PORT \
    --ctx-size $CTX -np $SLOTS -ctk $KV -ctv $KV -ngl 999 --jinja --reasoning-effort low -n 16384 \
    > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo -n "loading Qwen3.8-27B $MODEL (log: $LOG)"
  until healthy; do
    if ! our_pid >/dev/null; then echo; echo "llama-server exited during load:" >&2; tail -5 "$LOG" >&2; exit 1; fi
    echo -n "."; sleep 3
  done
  echo " ready ($MODEL, $SLOTS slot(s), $CTX tokens split across slots, $KV cache)."
  echo "Run ~/edwin/qwen, or open http://localhost:$PORT through ssh -L."
  ;;
stop)
  PID=$(our_pid) || { echo "no llama-server started by this script is running"; exit 0; }
  kill "$PID" && rm -f "$PIDFILE" && echo "stopped llama-server (pid $PID)"
  ;;
status)
  if healthy; then echo "running on :$PORT (pid $(our_pid || echo '?'))"; else echo "not running"; fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  ;;
*) echo "usage: $0 [start [--model q5|q6] [--slots N] [--ctx N] [--kv f16|q8_0]|stop|status]" >&2; exit 2 ;;
esac
