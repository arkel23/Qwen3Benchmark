# Qwen local models on server-3090 — agent/orchestrator usage notes

Setup: `qwen_3090_setup.sh` (this directory). Human-facing guide:
`qwen_3090_USAGE_HUMANS.md`. What's below is what actually mattered once running it for
real — read this before the next session that uses it.

## Status as of 2026-08-24 (27B benchmark)

Qwen3.8-27B benchmarked as an executor on the same two tasks the 9B/Sonnet ran, with
byte-identical prompts and fixtures (verified with `cmp`). Results, scoring and the
steering recipe: `qwen_3090_USAGE_HUMANS.md`. Hard numbers to reuse:

- **ctx 110000 OOMs.** KV cache alone requests 6880 MiB; weight buffer is 15718 MiB. On a
  fully free 24 GB card the real ceiling is ~98K. **Use `--ctx-size 49152`** → 19868 MiB
  total, 85 s load, and it matches what the 9B ran at (better comparability).
- Prefill ~1004 tok/s weighted, decode **37.2 tok/s** (very stable, 35.5–38.9 across all
  requests). 9B for reference: 1978 / 48.9.
- Per task: mean 315 s wall, ~30k tokens, 6–16 agentic turns (it self-verifies).
- **Reliability:** 2 of 8 attempts died with `Model stream ended after a tool result
  without visible progress` and produced no file; both passed on retry. Always confirm an
  artifact exists before scoring.
- **Do not run two Qwen Code sessions against one llama-server** if you need per-run
  timings — the `print_timing` lines interleave and byte-range attribution breaks.
- Contention: the box gets grabbed by other agents' jobs mid-load (the 17 GB read takes
  ~85–90 s, and two launches lost that race). `/tmp/launch_27b.sh` on the box waits for
  ≥20500 MiB free and auto-retries the OOM race.
- Benchmark harness (reusable): `$CLAUDE_JOB_DIR/tmp/qwen27b/` — `run_one.sh` (wraps a run
  and records log byte offsets + wall-clock + VRAM), `verify_task_a.py` (executes a
  candidate, self-diff-zero + tier-stat ground-truth checks, 8 fidelity markers),
  `verify_task_b.py`, `extract_timings.py` (per-run prefill/decode from a log byte range).
- **`verify_task_a.py` forces `--debugging` and `WANDB_MODE=disabled`.** Keep it that way:
  an early version omitted it and synced two junk runs to `nycu_pcs/KD_TGDA` (since
  deleted). Any harness that executes these scripts must suppress wandb.

## Status as of 2026-08-20 (evening update)

Validated end-to-end (Qwen Code -> llama-server -> tool_use -> file edit) with
Qwen3.5-4B, then used **Qwen3.5-9B for real steering-experiment work** across three
tasks of increasing difficulty (see `qwen_3090_USAGE_HUMANS.md` for results/recipe).
Qwen3.8-27B has since been fully benchmarked on the same tasks — see the 2026-08-24
section above; the once-open "full context production run" item is closed (110000 OOMs;
49152 is the working config).
Downloaded GGUFs in `$HF_HOME`: Qwen3.8-27B, Qwen3.5-9B, Qwen3.5-4B (all UD-Q4_K_XL).
No servers are left running.

## Day-to-day: start the server

```bash
ssh server-3090
export HF_HOME=/mnt/hdd10tb/edwin/hf
GGUF="$(find $HF_HOME/hub/models--unsloth--Qwen3.8-27B-GGUF -name 'Qwen3.8-27B-UD-Q4_K_XL.gguf')"
nohup /mnt/ssd2tb/edwin/projects/llama.cpp/build/bin/llama-server \
  -m "$GGUF" --host 127.0.0.1 --port 8080 --ctx-size 49152 -ngl 999 \
  > /mnt/hdd10tb/edwin/qwen_llama_server.log 2>&1 < /dev/null &
disown
```
(49152, not 110000 — the latter OOMs, see above. 19868 MiB resident, 85 s load.)
For the 9B (fits alongside other jobs; the workhorse for executor tasks):
swap the `find` for `models--unsloth--Qwen3.5-9B-GGUF -name 'Qwen3.5-9B-UD-Q4_K_XL.gguf'`
and use `--ctx-size 49152`.

Then from the client machine: `ssh -f -N -L 8080:127.0.0.1:8080 server-3090`,
and `qwen` (interactive) or `qwen -p "..."` (headless, see flags below).

**Already on server-3090?** Skip the tunnel entirely — llama-server and Qwen Code both
on localhost. Qwen Code needs Node, which is NOT on the box; install once via nvm (no
sudo), same as the WSL client: `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash`,
`nvm install 22`, `npm install -g @qwen-code/qwen-code@latest`, and the same
`~/.qwen/settings.json` pointing at `http://127.0.0.1:8080/v1`.

## Running the FGIR-ViT / TGDA / QuantizedASR-family repos (2026-08-20)

Adopted convention: **`python -m tools.<sub>.<script>` from the repo root** — no
editable installs needed (CWD puts the package root on `sys.path`). The previous
`pip install -e` installs of `tgda`/`fgir_vit` into `fgir_timm912` were removed.
Sibling imports inside `tools/` must be package-absolute for this to work
(`from tools.features_analysis.extract_features import ...`, not
`from extract_features import ...`) — already fixed in both repos' `compute_cka.py`.

## Caveats that actually bit us

- **This is a shared GPU.** `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
  **before every start** — never assume it's free, and never kill a PID that isn't your own
  llama-server. This session ran for hours alongside another agent's job that never finished.
- **Stopping llama-server: don't use `pkill -f "llama-server"`.** Its search pattern is
  itself a substring of the very shell command you're running it from (the `-f` flag matches
  full argv, including your own pkill invocation), so it self-matches and kills the wrapping
  shell/SSH session (hit this directly — exit code 255, no error message). Use instead:
  ```bash
  ps aux | grep '[b]in/llama-server' | awk '{print $2}' | xargs -r kill
  ```
- **Context size floor is ~30K tokens, not a few thousand.** Qwen Code's own system prompt +
  tool schemas alone consume ~30K tokens before your actual conversation starts. `--ctx-size
  16384` fails outright (`400: request exceeds available context size`). We ran the full
  agentic test at `--ctx-size 65536` successfully; `110000` is the plan's target for the 27B
  on a fully free card — go no lower than ~32768 even for quick checks.
- **Headless mode (`qwen -p ...`) needs `-y` and/or `--safe-mode`.** Qwen Code ships a
  built-in `computer_use__set_config` tool that (at least with these models) gets called
  unprompted; in non-interactive mode it can't get the approval it's asking for, and the
  model just retries the identical call until loop-detection aborts the run. Fix:
  ```bash
  qwen --safe-mode -p "..." -y -o text
  ```
  `--safe-mode` also disables extensions/hooks/MCP/QWEN.md, which is fine for a quick check
  but means it isn't testing your real customized setup — drop it once the base loop is
  confirmed working. **Normal interactive use (`qwen` with no `-p`) has not shown this
  problem** — the TUI just shows the approval prompt like any other tool call.
- **The GGUF lives on the spinning HDD (`/mnt/hdd10tb`), not the SSD.** Loading the 27B is
  disk-bound (confirmed via `iostat`: ~97% util on `sda` during load, process in `D`
  state) and takes 2-3 minutes just for the read — this is normal, not a hang. The small
  4B test model loads in seconds for comparison. Don't kill-and-retry a load that looks
  stuck at `{"error":{"message":"Loading model",...}}` before waiting several minutes.
- **`llama-server`'s default port is changing to `:9931` in a future llama.cpp release**
  (its own startup warning). We always pass `--port 8080` explicitly, so this is only a
  problem if that flag is ever dropped — don't rely on the default.
- **No auth on the server.** It's bound to `127.0.0.1` on purpose — always reach it through
  the SSH tunnel, never bind `--host 0.0.0.0`.

## What we actually measured (real numbers, not the plan's estimates)

- GGUF metadata reports the model as 27,320,697,856 params, 17,548,181,504 bytes on disk
  (~16.34 GiB) — matches the plan's HF-content-length estimate (16.35 GiB) almost exactly.
- At `--ctx-size 4096` (forced by sharing the card with the other job), actual GPU usage was
  17,020 MiB total — i.e. only ~280 MiB of overhead above raw weights at a near-zero context.
  The plan's ~1 GiB overhead + KV-cache-per-token math is untested at real scale (110K
  context) since the card was never fully free this session — treat the plan's ~106K-token
  context estimate as directional until confirmed with the full run.
- Confirmed working generation at that reduced config: ~19.3 tok/s output, ~56 tok/s prompt
  processing, correct and coherent answers, while sharing the card with a second ~4.7 GB job.

## Closed 2026-08-24 (was: "still open")

All four items are done — see the 2026-08-24 section at the top. Outcomes that corrected
the estimates recorded above: the ~106K-token context estimate was optimistic (110000
OOMs; weight buffer 15718 MiB + KV 6880 MiB at 110K exceeds 24 GB), VRAM at the working
`--ctx-size 49152` is **19868 MiB**, and full agentic tool-use at that context works —
six real executor runs, prefill ~1004 tok/s, decode 37.2 tok/s.
