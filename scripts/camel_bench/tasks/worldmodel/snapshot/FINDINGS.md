# Findings

One entry per result, including nulls, with the command that produced it.

## Stage 1a: the 288 wall-ball programs form 200 behavioural classes

Command: `python scripts/probe_sufficiency.py`.

Two programs are equivalent when their visible trajectories match under every probe action
sequence. Probes are open-loop: controller actions recorded on reference programs, plus uniform
random actions.

| Probe suite (controlled + random × ticks) | Classes | Classes split by a held-out suite (128 + 128 × 512, other seed) |
|---|---|---|
| 8 + 8 × 128 | 195 | 5 |
| 32 + 32 × 256 | 200 | 0 |
| 64 + 64 × 256 | 200 | 0 |
| 128 + 128 × 512 | 200 | 0 |

The default suite is 32 + 32 × 256. 88 of the 288 programs therefore have a behavioural twin, so
splits and scores are by class.

## Stage 1a: rule coverage of controller rollouts

Command: the coverage snippet in `claude_process.sh`.

Share of the 288 programs whose single controller rollout (seed = program index) contains at least
one paddle hit and at least one miss:

| Ticks | With a hit | With a miss | Median hits | Median misses |
|---|---|---|---|---|
| 128 | 0.986 | 0.889 | 2 | 3 |
| 256 | 1.000 | 0.983 | 5 | 7 |

Rollouts for data therefore use 256 ticks.

## Stage 1a: seen rollouts identify the program (gate 2 passes)

Command: `python scripts/identifiability.py`.

A task is a program with K seen controller rollouts of T ticks. It is uniquely identified when every
program that reproduces those rollouts under the recorded actions is in the true program's class.
Three seed sets over all 288 programs (864 tasks):

| K × T | Uniquely identified | Consistent classes, mean | Max |
|---|---|---|---|
| 4 × 128 | 0.920 | 1.08 | 3 |
| 4 × 256 | 0.971 | 1.03 | 2 |
| 6 × 256 | 0.986 | 1.01 | 2 |

Stage-1a data uses K = 4, T = 256. The gate required at least 0.90.

## E1 sanity run (pipeline check, not a result)

Command: `python scripts/train_e1.py --split combination --steps 300 --batch 8 --eval_seeds 1 --out <dir>`
on server-3090.

- 3.62M parameters; 166 train classes, 34 test classes; peak 2.61 GiB; 0.1 s per step.
- Loss fell from 6.47 to 1.79 in 300 steps.
- Test classes: top-1 equivalence 0.00; with execution filtering 0.03 (k = 8) and 0.12 (k = 64).
  The grammar prior with the same filtering reaches 0.06 (k = 8) and 0.32 (k = 64). The paddle-hit
  slot is 0.00 on test classes, whose hit rule (segment) appears in training only with bouncing
  walls.
- The grammar prior is strong at k = 64 because the family is small, so the gate compares top-1
  and k = 8.

## Environments

- server-3090: conda env `v2gc` — Python 3.12, numpy 2.2.6, pytest 8.3.5, torch 2.6.0+cu118, CUDA
  available.
- ed520: `/home2/video/edwin/support/envs/v2gc`, same pins, CUDA available on the RTX 2080 Ti.
  `/home2/video` free space went from 950 G to 942 G. `~` stayed at 27,320,912 KB before and after
  the install.

## E1 on the 2080 Ti: measured fit (ed520)

Command: `python scripts/train_e1.py --split combination --steps 100 --batch 16 --eval_seeds 1 --out <dir>`.

- 0.16 s per step at batch 16, peak 5.15 GiB, so the gate runs use ed520 (first rung of the ladder).
- Two concurrent runs would need about 10.3 GiB of 11 GiB, so runs are sequential.
- All 50 tests pass on ed520, including the GPU renderer check.
