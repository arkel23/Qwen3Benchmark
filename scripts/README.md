# scripts

- `llama_server.sh`, `qwen` — start, stop and check the llama.cpp server on camel, and open
  Qwen Code against it. See `../docs/local_models/LAUNCH_QWEN.md`.
- `qwen_3090_setup.sh`, `qwen_v100_setup.sh` — the setup commands as run on server-3090 and
  on camel.
- `camel_bench/` — the harness, tasks and graders of the model comparison in
  `../docs/camel_model_bench.md`.

Benchmark invocations are inline in the docs rather than wrapped in scripts, so each
number can be reproduced by copy-pasting a single command.
