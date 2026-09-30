# deprecated

Retired material. Nothing here is deleted — it is kept so a past decision can be
inspected or reversed.

## 2026-08-25 — `QwenLocal/`

Was a loose, un-version-controlled folder at `~/analysis/QwenLocal/` holding the notes on
running local Qwen models on server-3090. Merged into this repo because it covers the same
box and toolchain as the benchmarks, and because it was not backed up anywhere.

Its content now lives at:

| was | now |
|---|---|
| `qwen_3090_USAGE_AGENTS.md` | `docs/local_models/qwen_3090_USAGE_AGENTS.md` (plus a section on the Qwen3.8 `xhigh`-reasoning runaway and the context-exhaustion fix) |
| `qwen_3090_USAGE_HUMANS.md` | `docs/local_models/qwen_3090_USAGE_HUMANS.md` (byte-identical) |
| `qwen_3090_setup.sh` | `scripts/qwen_3090_setup.sh` (byte-identical) |

Use the copies under `docs/local_models/` and `scripts/` — they are the maintained ones.
To restore the old layout: `git mv deprecated/QwenLocal ../QwenLocal`.

## 2026-09-17 — `run_qwen.sh`

The first camel launcher: it started the llama-server, opened Qwen Code and stopped the
server when Qwen Code exited. Replaced by `scripts/llama_server.sh` (server kept running)
and `scripts/qwen` (opened as often as needed). Restore by copying it back to `~/edwin/`.

## 2026-09-18 — `qwen_generation_uploads/`

The two Qwen-across-generations figures the owner uploaded, made by an earlier agent whose code
was never saved. `qwen_generations/plot.py` now draws both from sourced data in
`qwen_generations/data.csv`; `qwen_generations/README.md` lists where the new figures differ.
