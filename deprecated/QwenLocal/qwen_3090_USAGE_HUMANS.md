# Local Qwen on the 3090 — human guide

How to run a local coding model on `server-3090` and get useful work out of it.
Agent-facing details and failure modes: `qwen_3090_USAGE_AGENTS.md`.

## Start everything (already SSH'd into the 3090)

```bash
export HF_HOME=/mnt/hdd10tb/edwin/hf
GGUF="$(find $HF_HOME/hub/models--unsloth--Qwen3.5-9B-GGUF -name 'Qwen3.5-9B-UD-Q4_K_XL.gguf')"
nohup /mnt/ssd2tb/edwin/projects/llama.cpp/build/bin/llama-server \
  -m "$GGUF" --host 127.0.0.1 --port 8080 --ctx-size 49152 -ngl 999 \
  > /mnt/hdd10tb/edwin/qwen_llama_server.log 2>&1 < /dev/null &
disown
qwen        # once the model finishes loading; config already points at localhost:8080
```

Check `nvidia-smi` first; never kill jobs that are not yours. The 27B needs a free card
(swap the model path; keep `--ctx-size 49152` — see the ceiling note below); the 9B
coexists with other jobs. Loading reads from the HDD — the 27B takes ~85 s, and other
jobs can claim the card during that window, so re-check afterwards.

From the WSL laptop instead: open `ssh -f -N -L 8080:127.0.0.1:8080 server-3090`
first, then `qwen` locally.

Stop the server:
```bash
ps aux | grep '[b]in/llama-server' | awk '{print $2}' | xargs -r kill
```
(Not `pkill -f llama-server` — it matches its own command line and kills your shell.)

## Which model

- **Qwen3.5-9B** — fast executor. Fits beside other GPU jobs (~5.5 GB), handled every
  well-specified task. Use when the work is mechanical and the spec is exact.
- **Qwen3.8-27B** — needs the card to itself (**19.9 GB** at ctx 49152). ~3× slower per
  task but qualitatively different: it investigates, self-verifies, and finds things the
  spec did not mention. Use when the task needs judgement rather than typing.
- Headless one-shots need `--safe-mode -p "..." -y`; interactive `qwen` does not.
- **Context ceiling for the 27B is ~98K, not 110K** — the KV cache alone wants 6.88 GB at
  110K and the load OOMs. 49152 is a safe default and leaves plenty above Qwen Code's
  ~30K floor.

## Steering recipe (2026-08-20 experiments, 27B data added 2026-08-24)

Real tasks on the TGDA/FGIR-ViT repos, run with the 9B and the 27B under several prompt
styles, with a Claude Sonnet 5 agent as the reference. The rules that fell out:

1. **Spell out the change for the 9B; you may brief the 27B.** The 9B matched Sonnet
   exactly on a one-line bug fix and on a two-file port whose steps were given precisely.
   It failed the same port when told only "make it work": it fixed the file in front of it
   and never discovered that a *second* file needed new arguments. **The 27B passed that
   same hands-off test**, found all three needed arguments, and justified each. So: the 9B
   executes, the 27B investigates. Cross-file discovery is still your job with the 9B; the
   27B can be trusted with it if you verify the result.
1b. **The spec is the first suspect, not the model.** Two "model failures" we recorded
   were prompt defects: the spec said "no wandb", so every model correctly omitted wandb,
   and it never named the font block, so most omitted that too. Re-read your spec before
   blaming output. Conversely, an under-specified spec is where the 27B shines — it filled
   the font gap on its own.
2. **State style rules, and show one good/bad example for anything non-trivial.** With
   no style guidance the 9B commented every step and wrote docstrings that narrate the
   spec. With the repo's rules quoted, and best with a short do/don't pair, output
   became close to indistinguishable from repo code.
3. **Name everything.** Exact filenames, arg names, key formats, insertion anchors
   ("after the `--fig_ext` line"). Every deviation we saw traced to a detail the prompt
   left open (file naming produced `_png` instead of `.png`; spec-adjacent keys instead
   of the requested ones).
4. **Verify by running, not reading.** Every accepted change was executed against real
   data before landing. The one bug that survived code review (wrong filename join) was
   caught only at run time.
5. **Trust boundary.** 9B + precise spec + your verification run: fine for mechanical
   edits, ports with known steps, single-file scripts against a fixed format. The 27B
   extends that to cross-file discovery and self-checked work, at ~3× the wall-clock and
   sole use of the GPU. Design decisions and anything spanning many files still belong to
   you or a frontier model.
6. **Budget for retries on the 27B.** 2 of 8 attempts died mid-loop with `Model stream
   ended after a tool result without visible progress`, leaving no artifact; both
   succeeded on a straight retry. The 9B never hit this. Script your runs so a retry is
   one command, and check that a file was actually produced before scoring a run.
7. **Style rules still pay off at every size.** Bare spec on the 27B produced 172 lines
   with 5 docstring markers; the same task with the repo's rules quoted came in at
   116–121 lines and a single docstring — tighter than the human-reviewed file that
   shipped. Neither size adds unrequested CLI args or features when told not to.

## Measured numbers

Same tasks, same byte-identical prompts and input fixtures for all three columns.

| | 9B local | **27B local** | Sonnet 5 agent |
|---|---|---|---|
| One-line bug fix | correct, all 3 styles | not re-run | correct; 45k tok, 15 s |
| New ~120-line analysis script | correct logic 3/3; style/naming flaws in 2 | **correct 3/3, no defects**; added repo font styling unprompted | correct, spec-exact; 52k tok, 71 s |
| Two-file port, steps given | exact diff, 24 s | exact diff, 95 s | — |
| Two-file port, "make it work" | **failed** (missed 2nd file) | **passed**, found all 3 args | passed, found all 3; 63k tok, 46 s |
| Prefill / decode | 1,978 / 48.9 tps | 1,004 / **37.2** tps | — |
| VRAM | ~5.5 GB | **19.9 GB** (needs the card alone) | — |
| Wall-clock per task | 24–154 s | 95–557 s (mean 315 s) | 15–71 s |
| Tokens per task | — | ~30k (mean; 182k over 6 runs) | 45–65k |
| Reliability | 6/6 first try | **2 of 8 attempts died** mid-loop, needed retry | 3/3 |

Decode speed halves when the GPU is shared; quality is unaffected. The 27B's decode rate
is remarkably stable (35.5–38.9 tps across every request).

### What the 27B did that the others did not

- **Found a real data bug nobody asked it to look for.** It reported that one input
  fixture's `cka_test` matrix was entirely NaN. Verified: 169/169 NaN. The 9B, Sonnet, and
  the orchestrator all missed it because everyone only exercised the `train` split. The
  cub CKA test-split data on the server is unusable and needs regenerating.
- **Fixed a style defect unprompted.** All three of its replication runs copied the repo's
  `rcParams` serif/Times block; the 9B and Sonnet had zero font handling in any run. On
  the corrected spec it used the repo's real `getattr(args, 'font_family', ...)` idiom —
  closer to house style than the version that actually shipped, which hardcoded them.
- **Exceeded the shipped solution on the port**, adding `--logits_layer` (which
  `selection_metrics.py` reads via `getattr` and whose own error message tells users to
  pass it) — still missing from the committed code.
- **Self-verifies by default**: it runs what it writes, checks the numbers, and reports
  discrepancies. That is where the 2–3× extra wall-clock and the 13–16 agentic turns go.

### Which to reach for

| Situation | Use |
|---|---|
| Exact spec, mechanical edit, card shared | 9B |
| Needs discovery, judgement, or you want the output checked | 27B |
| Latency matters, or the task is trivial | 9B or Sonnet |
| Card is busy | 9B only (27B will not fit) |

## Repo invocation convention

All the `~/projects` vision/ASR repos run scripts as `python -m tools.<dir>.<script>`
from the repo root — no `pip install -e` needed. See the agents doc for details.
