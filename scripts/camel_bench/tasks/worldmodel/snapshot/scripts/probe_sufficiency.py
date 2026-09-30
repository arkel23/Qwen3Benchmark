"""How large a probe suite separates every behaviourally different program in the family.
Run from the repo root: python scripts/probe_sufficiency.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pong.classes import equivalence_classes, probe_actions, replay_all
from pong.rules import all_programs

SUITES = [(8, 8, 128), (32, 32, 256), (64, 64, 256), (128, 128, 512)]


def main():
    programs = all_programs()
    held_out = replay_all(programs, probe_actions(ticks=512, controlled=128, random=128, seed=7))
    for controlled, random, ticks in SUITES:
        ids = equivalence_classes(programs, probe_actions(ticks=ticks, controlled=controlled, random=random))
        split = sum(
            not all(np.array_equal(held_out[m[0]], held_out[k]) for k in m[1:])
            for m in (np.flatnonzero(ids == c) for c in np.unique(ids))
        )
        print(f"probes {controlled}+{random}x{ticks}: {ids.max() + 1} classes, {split} split on held-out probes")


if __name__ == "__main__":
    main()
