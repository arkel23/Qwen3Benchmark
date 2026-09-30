"""Share of tasks whose seen rollouts identify the program up to behavioural equivalence.
A task is a program with K seen controller rollouts of T ticks. Run from the repo root: python scripts/identifiability.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pong.classes import consistent_programs, equivalence_classes, probe_actions
from pong.rules import all_programs
from pong.sim import simulate

SETTINGS = [(4, 128), (4, 256), (6, 256)]
SEED_SETS = 3


def main():
    programs = all_programs()
    ids = equivalence_classes(programs, probe_actions())
    for rollouts, ticks in SETTINGS:
        unique, sizes = [], []
        for seed_set in range(SEED_SETS):
            for i, program in enumerate(programs):
                seeds = 10_000 * (seed_set + 1) + i * rollouts + np.arange(rollouts)
                visible, actions = simulate([program] * rollouts, ticks, seeds=seeds)
                consistent = consistent_programs(programs, actions, visible)
                classes = np.unique(ids[consistent])
                unique.append(len(classes) == 1)
                sizes.append(len(classes))
        print(f"K={rollouts} T={ticks}: {np.mean(unique):.3f} of {len(unique)} tasks uniquely identified; "
              f"consistent classes mean {np.mean(sizes):.2f}, max {max(sizes)}")


if __name__ == "__main__":
    main()
