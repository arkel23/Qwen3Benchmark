# 11 — Which local model to serve on camel

Five candidates were screened on camel (1× V100 32 GB) to answer two questions: does any model
beat the deployed Qwen3.8-27B UD-Q4_K_XL on the work the team actually asks for, and how many
people can share the card at once. The best models then wrote real research code (section below).

All runs use llama.cpp `972d231` and the same sampling (temperature 0.6, top-p 0.95, top-k 20),
one request at a time unless the row says otherwise.

## Candidates

| id | model | size | placement | context used |
|---|---|---|---|---|
| M0 | Qwen3.8-27B UD-Q4_K_XL (served when the comparison ran) | 17.6 GB | all GPU | 229,376 |
| M1 | Qwen3.8-27B UD-Q5_K_XL | 20.9 GB | all GPU | 131,072 |
| M2 | Qwen3.8-27B UD-Q6_K_XL | 25.3 GB | all GPU, q8_0 KV cache | 131,072 |
| M3 | Ornith-1.5-35B-A3B Q5_K_M | 25.3 GB | all GPU, q8_0 KV cache | 131,072 |
| M4 | Qwen3.8-Flash-Next UD-IQ3_XXS | 82.0 GB | 36 layers of experts on CPU | 131,072 |

M2 and M3 need the quantized KV cache to reach 131k on this card. M4's expert offload was tuned
by measurement: 36 CPU layers gave 19.3 tok/s, against 15.0 at 44 layers and 14.7 at 32. Decode
speed is not monotonic in the offload level, because moving experts to the GPU adds PCIe traffic
per token.

## Screening tasks

`scripts/camel_bench/` holds the harness. Every grader was negative-tested before use: a correct
reference scores 1.0, a planted bug or an empty answer scores 0 (`selftest.py`, `0 failures`).

- **Code generation**, 12 tasks × 3 repeats: research utilities (edit distance, top-k accuracy,
  linear CKA, bootstrap CI, round-half-up, CSV → LaTeX, paired merge, EMA smoothing, argparse CLI,
  patch embedding, gradient accumulation, stratified split). Score is the share of hidden tests
  that pass.
- **Reading**, 6 documents of 7k to 96k tokens built from the team's own papers. Each has 10
  questions with an exact answer printed in the text, plus one 150-word summary scored on staying
  within the limit and covering five key facts.
- **Instruction following**, 5 paragraphs from our papers × 3 repeats: rewrite under rules
  (no sentence over 25 words, no semicolons or em-dashes, every number and LaTeX command kept).

## Screening results

| model | code | instructions | reading Q&A | summary |
|---|---|---|---|---|
| M0 Q4 | 0.969 | 0.987 | 1.000 | 0.667 |
| **M1 Q5** | **1.000** | **1.000** | **1.000** | 0.500 |
| M2 Q6 | 0.991 | 0.987 | 1.000 | 0.667 |
| M3 Ornith | 0.818 | 1.000 | 0.917 | 0.167 |
| M4 Flash-Next* | 0.975* | 1.000* | 1.000* | 0.750* |

\* Partial: Flash-Next ran the 12 code tasks once instead of three times, the 5 instruction tasks
once, and the four documents up to 29k tokens. The two ~95k-token documents were not run, because
it reads prompts at 14.5 tok/s and each would have taken about two hours. Its scores therefore rest
on 61 replies against 117 for every other model, and its reading score covers no long document.

No request failed for any model. The three 27B quants sit at the ceiling of these tasks, so the
gaps among them are one or two test cases and are not evidence of a real difference. Two results
are solid. Ornith is clearly weaker: it fails `round_half_up` on every attempt and loses reading
accuracy at every document length. And every model overruns the 150-word summary limit at least
twice, which is the only place instruction following visibly breaks.

## Speed

`llama-bench`, prompt of 512 and 128 generated tokens, at three context depths. The concurrent
rows use four server slots and one shared card.

| model | prompt tok/s (0 / 32k / 100k) | generate tok/s (0 / 32k / 100k) | 4 users, each | VRAM |
|---|---|---|---|---|
| M0 Q4 | 816 / 506 / 280 | 32.7 / 21.2 / 13.5 | 17.8 | 31,056 MiB |
| M1 Q5 | 849 / 512 / 282 | 28.7 / 20.5 / 12.3 | 15.6 | 27,666 MiB |
| M2 Q6 | 886 / 524 / 284 | 24.7 / 20.3 / 10.6 | 14.5 | 28,520 MiB |
| M3 Ornith | 803 / 609 / 440 | 99.5 / 81.6 / 59.0 | 50.3 | 25,578 MiB |
| M4 Flash-Next* | 7.0 / — / — | 3.7 / — / — | 2.9 | 20,742 MiB |

\* Flash-Next was measured at depth 0 only. During screening its server reported 14.5 tok/s of
prompt processing at 30k tokens and 5.6 tok/s of generation at 33k.

Four people can share the card. On any 27B each of them sees about 15 to 18 tok/s, which is faster
than reading speed; on Ornith, about 50. A 100k-token document costs about six minutes of prompt
reading on a 27B and under four on Ornith.

## Why Flash-Next does not run well here, and what it would take

Flash-Next answers as well as the 27B on the tasks it finished, so its problem on camel is memory,
not quality. Its weights are 81.96 GB (UD-IQ3_XXS, three shards) and llama.cpp maps them from
disk. camel has 84.3 GB of RAM, of which about 80 GB can serve as page cache, so the file alone is
97% of installed memory. With 36 of the 48 layers' experts on the CPU, 20.7 GB of the model sits on
the GPU, yet decoding still read from NFS continuously. The page cache grew past 73 GB without
settling. Sustained speed fell from 19.3 tok/s to 2.9-5.6 tok/s; the 19.3 was measured on the first
short prompt, while most experts were still untouched.

Running it without paging needs the whole file resident plus room for the process and the operating
system. **That is about 86 GB at the current settings, so 96 GB of RAM in practice, and 128 GB to
leave the usual page cache intact.** More CPU offload does not help, since it moves weights towards
RAM, and less offload is capped by the 32 GB card. The alternative within 84 GB is a smaller
quantization. UD-IQ1_S is 72.5 GB and would fit, but Unsloth reports its quality at 79.7% top-1
against 92.3% for its 4-bit build, which gives up the quality that is the reason to run this model.

## Writing real research code

The two leaders, M1 and M2, then wrote code for a live repository: `VideoToGameCode` at commit
`f48c90c`, frozen into `scripts/camel_bench/tasks/worldmodel/snapshot`. Two files were removed and
each model had to write one back:

- **W1, `render.py` in `pong`**: render visible states to RGB frames so that a frame determines the
  state it came from. Graded by the repo's own rendering test.
- **W2, `classes.py` in `pong`**: behavioural equivalence classes and consistency of a program with
  observed rollouts. Graded by the repo's tests and by whether the snapshot's `probe_sufficiency.py`
  reproduces the counts in its `FINDINGS.md` (195 classes with 5 split, then 200 with 0).

Each file was requested at two information levels, byte-identical across models: **blind** (goal
only) and **briefed** (the file's interface and the layout rules). Opus 5 wrote the reference
versions, 37 and 41 lines.

| | W1 blind | W1 briefed | W2 blind | W2 briefed |
|---|---|---|---|---|
| M1 Q5, one reply | 0.00 | 0.00 | 0.25 | 0.25 |
| M2 Q6, one reply | 0.00 | **1.00** | 0.25 | **1.00** |
| M1 Q5, with tools | — | **1.00** | — | — |
| M2 Q6, with tools | — | **1.00** | — | — |

Reading the failures matters more than the scores:

- **Blind never works.** Without the brief, both models invent a plausible frame layout, and the
  test that a frame determines its state fails. The equivalence-class file gets closer: it passes
  one of two tests and produces 196 classes with 4 split, against the true 195 and 5.
- **The near miss is the dangerous case.** M1's briefed `classes.py` ran, produced sensible output
  and reported 194 classes with 6 split, and 199 with 1 where the reference has 200 with 0. A
  result like that would pass a reading review and quietly change a paper's numbers. Only running
  the script against the recorded values catches it.
- **Tools close the gap.** Given Qwen Code and an hour, both models wrote a renderer that passes,
  in 54 and 41 lines against the reference's 37, with no style flag raised. In one reply the same
  task took six minutes and M1 failed it, on a NumPy broadcasting bug in its own code that one
  execution would have exposed.
- **Cost.** An hour of the card per agentic file, because these models sustain about 16 tok/s in
  long tool-use turns. A single reply costs three to six minutes.

A briefed run is implementation, not design: the brief carries the interface and the layout rules,
which is most of the design work. A 1.00 there means the model wrote a specified file correctly, not
that it designed the renderer. The blind column is the one that tests design, and both models fail it.

Each cell is one run, so treat the M1/M2 difference as weak. What is consistent across every run:
the brief is what makes the task possible, and execution is what makes the result trustworthy.

## What to serve

camel serves M1 Q5 by default and M2 Q6 on request; the other candidates are deleted.

- **Q5 for chat and most coding.** It ties or leads on every screening task, is faster than Q6
  (28.7 against 24.7 tok/s), and fits 196,608 tokens of context with an f16 cache.
- **Q6 for hard single-reply coding.** It wrote both briefed research files correctly where Q5 wrote
  neither. One run per cell, so this is a lead, not a measured gap.
- **Complex code still needs a brief, tools and execution.** Blind requests failed for both models,
  and a briefed Q5 reply produced a plausible wrong result. Run every generated file against known
  values before trusting it.
- **Flash-Next needs about 96 GB of RAM** to run without paging; camel has 84 GB.

`docs/local_models/LAUNCH_QWEN.md` lists the start commands and the measured settings for each
model.
