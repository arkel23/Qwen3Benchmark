"""Classes, class-based splits and task batches for learning programs from rollouts."""
import numpy as np

from pong.classes import equivalence_classes, probe_actions
from pong.rules import SLOTS, all_programs
from pong.sim import simulate

SPLITS = ("random", "combination")


class Family:
    """The enumerated programs, their behaviour classes and one canonical program per class."""

    def __init__(self):
        self.programs = all_programs()
        self.class_of = equivalence_classes(self.programs, probe_actions())
        canonical = {}
        for i, cls in enumerate(self.class_of):
            canonical.setdefault(int(cls), i)
        self.canonical = np.array([canonical[c] for c in range(len(canonical))])
        self.slot_index = np.array([p.index() for p in self.programs])
        self.index_of = {tuple(row): i for i, row in enumerate(self.slot_index)}

    def split(self, kind, seed=0):
        """Train and test class ids. `combination` holds out segment hits with wall wrap."""
        classes = np.arange(len(self.canonical))
        if kind == "random":
            test = np.random.default_rng(seed).choice(classes, size=len(classes) // 5, replace=False)
        else:
            reps = [self.programs[i] for i in self.canonical]
            test = classes[[p.hit_vy == "segment" and p.wall == "wrap" for p in reps]]
        return np.setdiff1d(classes, test), np.sort(test)

    def batch(self, classes, rollouts, ticks, rng):
        """Rollouts for one canonical program per class: visible (B, K, T + 1, 4), actions (B, K, T), heights (B,)."""
        programs = [self.programs[self.canonical[c]] for c in classes]
        seeds = rng.integers(2**31, size=len(programs) * rollouts)
        visible, actions = simulate([p for p in programs for _ in range(rollouts)], ticks, seeds=seeds)
        b = len(programs)
        heights = np.array([p.paddle_height for p in programs])
        return visible.reshape(b, rollouts, ticks + 1, 4), actions.reshape(b, rollouts, ticks), heights

    def targets(self, classes):
        return self.slot_index[self.canonical[classes]]

    def class_of_slots(self, slots):
        """Class id for each row of slot indices (N, 6)."""
        return self.class_of[[self.index_of[tuple(row)] for row in slots]]


SLOT_SIZES = [len(values) for values in SLOTS.values()]
