# VideoToGameCode

Can a model that watches a game's video write the code of that game? This repo tests it on a family
of Pong variants on an integer lattice. Each variant is a rule program; a model sees rendered
rollouts (frames and the player's actions) and must output a program that behaves identically.

## Layout

| Path | Contents |
|---|---|
| `pong/rules.py` | The rule family: slots, the 288 stage-1a programs, canonical program text, and compilation to a Python `Game` class |
| `pong/sim.py` | The batched integer simulator and scripted controller; closed-loop rollouts and open-loop replay |
| `pong/render.py` | Rendering of visible states to RGB frames at any cell size |
| `pong/classes.py` | Behavioural equivalence classes from probe action sequences, and the programs consistent with observed rollouts |
| `scripts/probe_sufficiency.py` | Measures how large a probe suite separates every behaviourally different program |
| `tests/` | Engine tests |
| `FINDINGS.md` | One entry per result, with the command that produced it |

## Setup and tests

```bash
conda create -y -n v2gc python=3.12 && conda activate v2gc
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
python -m pytest -q tests
```

On ed520 the environment is created by prefix under `/home2/video/edwin/support/envs/v2gc`, with
package and model caches redirected there.
