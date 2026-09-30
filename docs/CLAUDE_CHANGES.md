# Changes log

Dated record of substantive changes and corrections.

## 2026-09-07 — documents to the current standard: moved history

Dates, history and finished items removed from the current-state documents, verbatim, with
the file and line they came from. Emphasis-only edits are not listed.

### From docs/model_trial.md:18

```
but only after I fixed a `token_type_ids` kwargs bug that crashed its own benchmark
```

### From docs/model_trial.md:43-45

```
Both T1 failures were **path/kwargs bugs in the candidates' own harness code**, not in
their core idea. Two of my invocation attempts were also wrong (missing `--config`, a
bad patch indent) — those are mine, and are not counted against the candidates.
```

### From docs/model_trial.md:84-89

```
But note the
27B was competitive on *quality* where it finished: its T3 is the shortest correct
solution of the six (199 L), and its blind T1 went furthest conceptually — it wrote its
own Triton ternary GEMM rather than reaching for GGUF. That approach stores int8 codes
(1 byte/weight), so it captures ~2× bandwidth savings rather than the 8× packed 2-bit
gives, and it was never executed.
```

### From docs/model_trial.md:100-101

```
An earlier conclusion that this was a 27B capability ceiling
   was **wrong** and is retracted here.
```

## 2026-09-18 — camel serves Q5 and Q6

- camel: `llama_server.sh --model q5|q6` (q5 default); 27B Q4, Qwen3.5-9B/4B, Flash-Next and
  Ornith deleted (478 → 603 GB free). `docs/local_models/LAUNCH_QWEN.md` is the operator sheet;
  `docs/camel_model_bench.md` gained "What to serve".
- Stopping the server by PID file replaced `pkill -u $USER`, which on a shared account would
  stop other people's servers.
- `qwen_generations/` rebuilds two uploaded figures from sourced data. Differences from the
  uploads: the Qwen1.5→2 and 3.5 drivers are architecture plus data, the GPQA series is fixed at
  27–32B dense (the upload mixed 235B, 397B, 35B-A3B and Max), and Qwen3.7 is dropped.

## 2026-09-23 — writing-rules review of the docs

- `qwen_v100_USAGE.md` no longer repeats the start, chat and session commands; it points to
  `LAUNCH_QWEN.md` and keeps only what the sheet lacks.
- `LAUNCH_QWEN.md` said the one-reply file failed where only q5 failed and q6 passed;
  `docs/camel_model_bench.md` had long sentences and a stale "(deployed)" label.
- The V100 speed rows name their build as d59d455 rather than "same build".
