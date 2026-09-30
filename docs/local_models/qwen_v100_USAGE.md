# Local Qwen on camel (V100 32 GB)

camel runs llama.cpp build `972d231`, Qwen Code, and two quantizations of Qwen3.8-27B:
UD-Q5_K_XL (default) and UD-Q6_K_XL. Which model to serve, and how many people can share the
card, is measured in `../camel_model_bench.md`; the start commands and settings are in
`LAUNCH_QWEN.md`. Setup commands: `../../scripts/qwen_v100_setup.sh`. The steering recipe in
`qwen_3090_USAGE_HUMANS.md` applies unchanged; its model choice is for the 3090 and does not.

## Machine

1× Tesla V100 32 GB (sm_70, driver 535, CUDA ≤ 12.2), 40 threads, 78 GiB RAM, CentOS 7
(glibc 2.17). No sudo. The login shell is tcsh; both scripts run from it directly.

| path | contents |
|---|---|
| `/edahome/pcslab/pcs05/edwin/projects/llama.cpp` | shallow checkout at `972d231`, built in `build/` |
| `~/.cache/huggingface/hub` | the Q5 and Q6 GGUFs (46 GB), in the default Hugging Face cache |
| `/edahome/pcslab/pcs05/edwin/tools` | Node 22 (glibc-2.17 build) with Qwen Code; the micromamba binary |
| `/edahome/pcslab/pcs05/edwin/logs` | `llama_server.log` and `llama_server.pid` |
| `/edahome/pcslab/pcs05/edwin/llama_server.sh` | starts, stops or checks the server (copy of `../../scripts/llama_server.sh`) |
| `/edahome/pcslab/pcs05/edwin/qwen` | opens Qwen Code against the server (copy of `../../scripts/qwen`) |
| `~/.qwen/` | Qwen Code settings and instruction files (copy in `camel_dot_qwen/`) |
| conda env `llamacpp-build` | gcc 12, CUDA toolkit 12.2, cmake, `hf`; for building only |

All paths are on NFS. A cold Q5 load reads 21 GB over it; a warm load takes ~10 s.

## Start, use, stop

Every command for starting, stopping, chatting and running Qwen Code is in `LAUNCH_QWEN.md`. Two
behaviours are not listed there. `llama_server.sh` refuses to start when the GPU has under
31,500 MiB free, and reports an already running server instead of starting a second one. `qwen`
passes its arguments to Qwen Code, for example
`~/edwin/qwen --approval-mode plan -o text -p "..." < /dev/null`, and `qwen sessions list` lists
the sessions stored for the current directory.

## Chat interface for other people

The chat interface is llama-server's own web page, reached through an SSH tunnel as
`LAUNCH_QWEN.md` shows. It has no user accounts.

Open WebUI, which adds accounts and server-side history, does not install on camel: it pins
`onnxruntime==1.26.0`, whose newest build for this machine's glibc 2.17 is 1.16.3. Run it on a
machine with a current Linux instead, pointed through the same tunnel at camel's server, which also
keeps its accounts and chat history off this shared account.

## Instructions for Qwen

Qwen Code reads its instructions from `~/.qwen/`.

| file | loaded |
|---|---|
| `AGENTS.md` | every session: machine, paths, envs, verification, change and git rules |
| `~/.qwen/rules/writing.md` | when a `.tex`, `.bib`, README, `docs/` or `paper/` file is opened (`paths:` frontmatter) |
| `~/.qwen/guides/code_repos.md` | read by the model when working in `edwin/projects/*`, as `AGENTS.md` instructs |
| `~/.qwen/guides/analysis_repos.md` | read by the model when working in `edwin/analysis/*` |

Every `.md` in `~/.qwen/rules/` without `paths:` loads in every session, so the
directory-conditional guides live outside it. `settings.json` lists `~/.qwen/guides` in
`context.includeDirectories`; without it each read of a guide asks for permission. An `@`
at the start of a word in these files imports that path, so they name paths without one.

In tests, a plain folder loaded `AGENTS.md` only. Opening a paper's `main.tex` injected the
writing rules. Two ordinary questions in `edwin/projects/FGIR-ViT` both read
`~/.qwen/guides/code_repos.md` and the repo's `CLAUDE.md` unprompted.

## Context size

This table is for the UD-Q4_K_XL file. camel holds Q5 and Q6, whose limits are in
`LAUNCH_QWEN.md`.

| config | loads | VRAM |
|---|---|---|
| ctx 262144, default 4 slots | no (OOM on the recurrent-state cache) | — |
| ctx 196608, default 4 slots | yes | 29,420 MiB |
| ctx 262144, `-np 1` | no (a 150 MiB state buffer fails) | — |
| **ctx 229376, `-np 1`** | **yes** | **31,050 MiB** |

`-np 1` matters: the default four slots allocate four recurrent-state caches. At
229376 a 147,815-token prompt ran without OOM and recalled a detail from its start.
VRAM peaked at 31,248 MiB. That context is 2.3 times the largest that fits on the 3090, 98,304
tokens, and raises the 3090's budget of about 2,000 to 2,500 lines of source per agentic task
accordingly.

## Speed

`llama-bench -ngl 999 -p 512 -n 128 -r 2`, llama.cpp `d59d455` and the UD-Q4_K_XL GGUFs on both cards, each
card otherwise idle. Machine-readable: `../../results/qwen_local_bench.csv`.

| model | test | 3090 t/s | V100 t/s | V100 / 3090 |
|---|---|---|---|---|
| Qwen3.8-27B | pp512 | 1363.06 | 826.51 | 0.61 |
| Qwen3.8-27B | tg128 | 41.63 | 33.57 | 0.81 |
| Qwen3.5-9B | pp512 | 3995.06 | 2380.03 | 0.60 |
| Qwen3.5-9B | tg128 | 119.86 | 91.10 | 0.76 |

The 27B rows use flash attention on; with it off, both cards are within 1.5%. In
server use the 27B decodes at ~30 t/s on short Qwen Code turns. At 147,815 tokens of
context, prefill averaged 349 t/s (424 s) and decode fell to 6.9 t/s. Long contexts fit,
but a full one costs minutes of prefill per uncached request.

## Caveats

- **`qwen -p` reads stdin.** Piping a script into `bash -s` that calls `qwen` lets it
  consume the rest of the script. Give it `< /dev/null` or run it from a file.
- **Shared account.** `pcs05` and its home, including `~/.qwen`, are shared with other
  people. Only kill a `llama-server` PID you started.
- Headless flags, the stop command, and `--reasoning-effort` behave as in
  `qwen_3090_USAGE_AGENTS.md`. The "Reply with exactly: OK" probe returns in 22 tokens.
