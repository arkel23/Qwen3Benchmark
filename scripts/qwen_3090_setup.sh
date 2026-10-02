#!/bin/bash
# qwen_3090_setup.sh -- one-time log of commands run to set up Qwen3.8-27B as a local
# coding-agent backend on server-3090 (llama.cpp/llama-server + Qwen Code harness).
# Usage notes / caveats hit while running this: ../docs/local_models/qwen_3090_USAGE_AGENTS.md.
# Convention: each block dated, commented out once run successfully (per ~/.claude/CLAUDE.md).
set -euo pipefail

REMOTE=server-3090

# --- 2026-08-20: build llama.cpp with CUDA support (no sudo; qwen3_5 arch needs current main) -- DONE
# ssh "$REMOTE" '
#   source /mnt/ssd2tb/miniconda3/etc/profile.d/conda.sh
#   conda create -n llamacpp-build -c conda-forge -c nvidia cmake ninja cuda-toolkit=13.0 -y
#   conda activate llamacpp-build
#   git clone https://github.com/ggml-org/llama.cpp /mnt/ssd2tb/edwin/projects/llama.cpp
#   cd /mnt/ssd2tb/edwin/projects/llama.cpp
#   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF
#   cmake --build build --config Release -j$(nproc)
# '

# --- 2026-08-20: download Qwen3.8-27B Unsloth Dynamic v3.0 Q4_K_XL GGUF into existing HF cache -- DONE
# ssh "$REMOTE" '
#   source /mnt/ssd2tb/miniconda3/etc/profile.d/conda.sh
#   conda activate asr
#   export HF_HOME=/mnt/hdd10tb/edwin/hf
#   hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf
# '

# --- 2026-08-20: small Qwen3.5-4B GGUF for quick pipeline smoke tests (no need to wait on the -- DONE
# --- 27B's disk-bound load, or on full GPU headroom, to sanity-check the tool-use loop) ---
# ssh "$REMOTE" '
#   source /mnt/ssd2tb/miniconda3/etc/profile.d/conda.sh
#   conda activate asr
#   export HF_HOME=/mnt/hdd10tb/edwin/hf
#   hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf
# '

# --- start llama-server (on demand -- not a permanent daemon). Full production config: ---
# --- ctx-size 49152 (110000 OOMs: KV 6880 MiB + weights 15718 MiB > 24 GB). Needs a free card. At time ---
# --- of writing the card was shared with another agent's job, so this was run at a reduced ---
# --- --ctx-size 4096 for a basic-inference check only, then stopped again. Always check ---
# --- nvidia-smi first; never touch a PID that isn't your own llama-server. ---
ssh "$REMOTE" '
  export HF_HOME=/mnt/hdd10tb/edwin/hf
  GGUF="$(find $HF_HOME/hub/models--unsloth--Qwen3.8-27B-GGUF -name "Qwen3.8-27B-UD-Q4_K_XL.gguf")"
  nohup /mnt/ssd2tb/edwin/projects/llama.cpp/build/bin/llama-server \
    -m "$GGUF" --host 127.0.0.1 --port 8080 --ctx-size 49152 -ngl 999 \
    > /mnt/hdd10tb/edwin/qwen_llama_server.log 2>&1 < /dev/null &
  disown
'
# Verify (loading is disk-bound off the HDD, allow a couple minutes for the 27B):
#   ssh "$REMOTE" "curl -s http://127.0.0.1:8080/v1/models"
# Stop (NOT `pkill -f llama-server` -- it self-matches this very command's argv and kills the
# wrapping shell/SSH session; hit this directly during testing):
#   ssh "$REMOTE" "ps aux | grep '[b]in/llama-server' | awk '{print \$2}' | xargs -r kill"

# --- 2026-08-20: local (WSL) client -- tunnel + Node via nvm (no sudo) + Qwen Code -- DONE ---
# Tunnel (run in its own terminal, kept open; -f/-N above backgrounds it instead if preferred):
#   ssh -L 8080:127.0.0.1:8080 -N server-3090
# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# export NVM_DIR="$HOME/.nvm"
# [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
# nvm install 22
# npm install -g @qwen-code/qwen-code@latest

# --- 2026-08-20: point Qwen Code at the tunneled llama-server (persistent config) -- DONE ---
# mkdir -p ~/.qwen
# cat > ~/.qwen/settings.json <<'JSON'
# {
#   "modelProviders": { "openai": [ { "id": "local-qwen", "name": "server-3090 llama.cpp",
#     "baseUrl": "http://127.0.0.1:8080/v1", "envKey": "OPENAI_API_KEY" } ] },
#   "env": { "OPENAI_API_KEY": "unused" },
#   "security": { "auth": { "selectedType": "openai" } },
#   "model": { "name": "local-qwen" }
# }
# JSON

# --- verification: interactive `qwen` in a real project dir. For a headless one-shot check, ---
# --- --safe-mode and/or -y are required or a built-in computer_use tool call can stall on an ---
# --- approval it can't get non-interactively and trip loop detection (see qwen_3090_USAGE_AGENTS.md): ---
#   qwen --safe-mode -p "read a file and describe it" -y -o text
# ssh "$REMOTE" nvidia-smi   # record actual VRAM use vs the plan's ~16.35GiB weights + KV estimate

# --- 2026-08-20 (afternoon): steering experiments -- DONE. Qwen3.5-9B GGUF downloaded to
# --- $HF_HOME (hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf); editable
# --- installs of tgda/fgir_vit removed in favor of `python -m tools....` from repo root.
# --- Results and recipe: qwen_3090_USAGE_HUMANS.md / qwen_3090_USAGE_AGENTS.md.

# --- 2026-08-24: Qwen3.8-27B executor benchmark -- DONE. Server run at --ctx-size 49152
# --- (110000 OOMs: KV 6880 MiB + weights 15718 MiB > 24 GB). Six runs of the same tasks the
# --- 9B/Sonnet did, byte-identical prompts/fixtures. Harness + raw outputs left in the job
# --- scratch dir; results and steering recipe in qwen_3090_USAGE_HUMANS.md.
