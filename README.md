# Qwen3Benchmark: benchmarks and setup of local Qwen 3 models

Lab notes on running Qwen 3 models locally on two GPU machines: server-3090 (one RTX 3090, 24 GB)
and camel (one V100, 32 GB). They cover how to set the models up on each machine, how fast they
run, which model to serve, and how a local model compares with a cloud model when writing research
code. The operator sheet for camel is `docs/local_models/LAUNCH_QWEN.md`.

On server-3090, `qwen_generations/plot.py` runs in the `asr` conda env, which has the packages in
`requirements.txt`, and llama.cpp is built in the `llamacpp-build` env. The `camel_bench` harness
runs on camel and needs only the Python standard library; camel's setup is in
`docs/local_models/qwen_v100_USAGE.md`.

## Layout

| path | contents |
|---|---|
| `docs/camel_model_bench.md` | which local model to serve on camel: five candidates screened, the best two writing real research code |
| `docs/model_trial.md` | tiered model trial: Qwen3.8-27B vs Sonnet 5 on one research-code task |
| `docs/local_models/` | running local Qwen models on server-3090 and on camel; `qwen_3090_USAGE_HUMANS.md` compares the 9B, the 27B and Sonnet 5 on a code port |
| `results/*.csv` | the measured numbers, machine-readable; the `camel_*` files hold the model comparison |
| `scripts/` | the exact commands used, plus `qwen_3090_setup.sh`, `qwen_v100_setup.sh` and `camel_bench/` |
| `qwen_generations/` | Qwen benchmark scores across generations, with sources, and the script that plots them |
| `requirements.txt` | the pinned packages of `qwen_generations/plot.py` |
| `deprecated/` | retired files, each with a note in `deprecated/README.md` |
