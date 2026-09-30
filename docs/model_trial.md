# Tiered model trial: Qwen3.8-27B vs Sonnet 5 on one research-code task

Background: the task comes from a speech-model codebase, QuantizedASR. Its ternary (1.58-bit)
decoder ran no faster and was no smaller than the fp16 one, because its ternary layer is a
training layer that keeps bf16 weights and quantizes them on every forward pass. The known
fix is to ternarize the weights for real and export the decoder to a packed GGUF file for
llama.cpp. A local model (Qwen3.8-27B on server-3090) and a cloud model (Claude Sonnet 5)
were each asked to find and implement that fix.

Same task both models: "make the ternary decoder actually faster/smaller than the fp16
one; deliver a standalone file that builds via the repo's own `build_q2a`." Three
information levels, byte-identical prompts and input fixtures per tier.

- **T1 blind** — goal + the two config names. No pointers, no conclusions.
- **T2 pointers** — adds "look at these files / these repos", no conclusions.
- **T3 full brief** — adds the distilled findings (fake-quant root cause, the
  `prequantized`-is-not-ternary trap, the absmean formula, the GGUF route).

Graded by executing every candidate on the real box, not by reading it.

## Results

| model | tier | wall | script | ternarization | **what happened when executed** |
|---|---|---|---|---|---|
| 27B | T1 blind | 3600 s (timeout) | 388 L | yes | **runs: 32.6 tok/s GPU, 2.00x smaller** — but only after a `token_type_ids` kwargs bug that crashed its own benchmark was fixed by the grader; FINDINGS.md never written |
| 27B | T2 pointers | 706 s | 251 L | **missing** | **exported 0/7 ternary (dense); PPL 31,218** |
| 27B | T3 brief | 685 s | 199 L | yes | **exported 7/7 ternary — correct** |
| Sonnet | T1 blind | ~240 s | ~210 L | yes | **packs to int2 (1.72x vs baseline) then crashes** before printing throughput, on a config-path bug in its own code |
| Sonnet | T2 pointers | ~247 s | 171 L | **missing** | **crashed: `'NoneType' has no attribute 'to_dict'` — no checkpoint** |
| Sonnet | T3 brief | ~232 s | 219 L | yes | **exported 7/7 ternary — correct** |

### The two T1s, measured rather than assumed

Both blind runs found the root cause and derived the ternarization, and both went for a
GPU solution instead of GGUF — but neither is deployable as delivered:

- **27B T1** wrote its own Triton ternary GEMM. It works: 32.6 tok/s on the 3090.
  For scale, on the same GPU the reference numbers are fp16 219, `Q2_0` 83.6,
  `TQ2_0` 39.5 tok/s. So its hand-written kernel is slower than the worst stock
  ternary format. Its size report explains why: it stores int8 codes, 1 byte/weight
  → 1585.4 MB vs 3170.9 MB fp16 = 2.00x, against 8.09x for packed TQ2_0. The idea was
  right (read codes directly, never dequantize); the packing left 4x on the table.
  Caveat: measured via HF `generate()`, not `llama-bench`, so not strictly comparable.
- **Sonnet T1** used `torchao` int2. It reached the packing stage —
  `[baseline BitNetLinear] 3439.6 MB` → `[fast int2 packed] 2002.9 MB`, 1.72x — then
  died loading its comparison config. 2002.9 MB is far above what true 2-bit packing
  should give (~400 MB), so its "int2" layout is not achieving 2 bpw either. No
  throughput was ever produced.

Both T1 failures were path/kwargs bugs in the candidates' own harness code, not in their
core idea. Two of the grader's invocation attempts were also wrong (missing `--config`, a
bad patch indent) and are not counted against the candidates.

All six import cleanly and expose a working CLI. That is exactly why static review is
not enough here.

## The headline: partial pointers were worse than none — in BOTH models

T2 gave file pointers (`bitnet_convert.py`, `onebitllms/layers/bitnet.py`, …) without
conclusions. Both models followed them to `detect_weight_format()` /
`unpack_model_to_linear()` and trusted those helpers — which, on the `prequantized`
revision, report `packed_bitnet` and then unpack zero layers, silently. Neither model
added the absmean ternarization. Blind T1, with nothing to trust, derived it from first
principles in both models.

Two independent models, same failure, same cause. Handing a model a pointer to a helper
that silently no-ops is worse than handing it nothing.

## What the silent failure actually costs

The 27B's T2 export ran to completion, produced a valid checkpoint, converted to GGUF and
loaded fine. Its weights were dense (5256, 4903, 4886 unique values per layer-0
tensor, vs 3 for a correct export). Pushing that through the same TQ2_0 pipeline:

| export | PPL |
|---|---|
| correct (ternarized, then TQ2_0) | **8.8247** |
| T2 trap (dense, then TQ2_0) | **31217.97** |

**~3,500× worse.** Quantizing dense bf16 weights into a 3-value-per-block format
destroys the model — but every step succeeds and the artifact runs. Nothing short of
running it and measuring would have caught this.

## Cost

| model | per completed tier |
|---|---|
| Sonnet 5 | ~4 min, both deliverables |
| Qwen3.8-27B | 11–12 min (T2/T3); T1 blew the full 60 min cap and never wrote FINDINGS.md |

~3× slower on the tiers it finished, and it failed the one Sonnet completed. The 27B was
competitive on quality where it finished. Its T3 is the shortest correct solution of the
six (199 L), and its blind T1 went furthest conceptually — it wrote its own Triton ternary
GEMM rather than reaching for GGUF. That approach stores int8 codes (1 byte/weight), so it
captures ~2× bandwidth savings rather than the 8× packed 2-bit gives.

## Harness lessons (cost more time than the task)

1. **`--reasoning-effort low` + `--jinja`.** Qwen3.8's chat template defaults to `xhigh`;
   unbounded thinking produced a 15,263-token single response and no deliverable.
   `--reasoning-budget` is the wrong lever — it truncates mid-stream.
   (https://huggingface.co/Qwen/Qwen3.8-27B/discussions/113)
2. **Context, not capability.** At `--ctx-size 49152` the run died with
   `Context size has been exceeded` at n_tokens=46025: ~30k of Qwen Code system prompt +
   tool schemas plus ~18k of source files. `--ctx-size 98304 -np 1` (22.5 GB, near the
   24 GB ceiling) fixed it.
3. Qwen Code sends its own `max_tokens`, so the server's `-n` is not binding.
4. Practical budget on 24 GB: ctx ~98k minus ~30k harness overhead, and input should be
   ~a third of the remainder → ~2,000–2,500 lines of source per agentic task.
