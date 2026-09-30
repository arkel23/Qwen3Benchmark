#!/bin/bash
# qwen_v100_setup.sh -- commands run on camel (1x V100 32 GB, CentOS 7, no sudo) to set up the
# same local Qwen stack as qwen_3090_setup.sh. Run each block in bash on camel (login shell is tcsh).
# Usage notes and measurements: ../docs/local_models/qwen_v100_USAGE.md.
set -euo pipefail

E=/edahome/pcslab/pcs05/edwin

# --- 2026-09-18: upgraded to master 972d231 (adds Qwen3.8-Flash-Next); configure the build at the
# --- final path, because its libraries are linked by absolute path and a moved build cannot load -- DONE
# cd $E/projects/llama.cpp && git fetch --depth 1 origin 972d2313bc0bf0a45f634f77d95c9fb03aeab12c
# git checkout -q FETCH_HEAD && cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
#   -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j 20

# --- 2026-09-17: llama.cpp at the 3090's build d59d455fd (a full clone dies with early EOF) -- DONE
# mkdir -p $E/projects/llama.cpp && cd $E/projects/llama.cpp
# git init -q && git remote add origin https://github.com/ggml-org/llama.cpp
# git fetch --depth 1 origin d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd && git checkout -q FETCH_HEAD

# --- 2026-09-17: build env via static micromamba (conda 23.5's classic solver stalls); driver 535 caps CUDA at 12.2 -- DONE
# mkdir -p $E/tools/micromamba && cd $E/tools/micromamba
# curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar xj bin/micromamba
# MAMBA_ROOT_PREFIX=$E/tools/micromamba/root ./bin/micromamba create -y \
#   -p /edahome/pcslab/pcs05/miniconda3/envs/llamacpp-build -c conda-forge -c nvidia --override-channels \
#   cmake ninja "cuda-toolkit=12.2" "gcc=12" "gxx=12" "sysroot_linux-64=2.17" huggingface_hub

# --- 2026-09-17: build for sm_70 -- DONE
# source /edahome/pcslab/pcs05/miniconda3/etc/profile.d/conda.sh && conda activate llamacpp-build
# cd $E/projects/llama.cpp
# cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
# cmake --build build --config Release -j 20

# --- 2026-09-17: the three GGUFs at the revisions used on server-3090 -- DONE
# HF_HOME unset: camel uses the default ~/.cache/huggingface
# hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --revision 27af057ecb382ddfea5d12837360a8980560e3ed
# hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --revision 3885219b6810b007914f3a7950a8d1b469d598a5
# hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf --revision e87f176479d0855a907a41277aca2f8ee7a09523

# --- 2026-09-18: Q5 (default) and Q6 of Qwen3.8-27B, the two models served -- DONE
# hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q5_K_XL.gguf Qwen3.8-27B-UD-Q6_K_XL.gguf \
#   --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502

# --- 2026-09-18: deleted the models not served (27B Q4, Qwen3.5-9B/4B, Flash-Next, Ornith) -- DONE
# removed their files from ~/.cache/huggingface/hub; free space 478 -> 603 GB

# --- 2026-09-17: Node 22 glibc-2.17 build (official builds need glibc 2.28) + Qwen Code 0.24.0 -- DONE
# cd $E/tools
# curl -sL https://unofficial-builds.nodejs.org/download/release/v22.23.2/node-v22.23.2-linux-x64-glibc-217.tar.xz | tar xJ
# export PATH=$E/tools/node-v22.23.2-linux-x64-glibc-217/bin:$PATH
# npm install -g @qwen-code/qwen-code@latest   # installed 0.24.0
# ~/.qwen/settings.json: same content as in qwen_3090_setup.sh, baseUrl http://127.0.0.1:8080/v1

# --- start: ~/edwin/llama_server.sh, then ~/edwin/qwen (copies in this directory) ---
