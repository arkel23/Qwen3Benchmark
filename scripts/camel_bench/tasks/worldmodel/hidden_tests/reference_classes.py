"""Behavioural equivalence classes and the identifiability ceiling over the enumerated program family.

Two programs are treated as equivalent when they produce the same visible trajectories under every probe action
sequence. Probes are open-loop: controller actions recorded on reference programs, plus uniform random actions.
"""
import hashlib

import numpy as np

from pong.rules import all_programs
from pong.sim import simulate


def probe_actions(ticks=256, controlled=32, random=32, seed=0):
    """Fixed probe action sequences (controlled + random, ticks), identical for every program."""
    programs = all_programs()
    rng = np.random.default_rng(seed)
    references = [programs[i] for i in rng.integers(len(programs), size=controlled)]
    _, recorded = simulate(references, ticks, seeds=rng.integers(2**31, size=controlled))
    return np.concatenate([recorded, rng.integers(-1, 2, size=(random, ticks))])


def replay_all(programs, actions):
    """Visible trajectories (P, A, T + 1, 4) of every program under every action sequence."""
    tiled = [p for p in programs for _ in range(len(actions))]
    visible, _ = simulate(tiled, actions.shape[1], actions=np.tile(actions, (len(programs), 1)))
    return visible.reshape(len(programs), len(actions), *visible.shape[1:])


def equivalence_classes(programs, actions):
    """Class id per program (ids numbered by first occurrence in `programs`)."""
    trajectories = replay_all(programs, actions)
    digests = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in trajectories]
    ids = {}
    return np.array([ids.setdefault(d, len(ids)) for d in digests])


def consistent_programs(programs, seen_actions, seen_visible):
    """Boolean mask over `programs`: which reproduce `seen_visible` (K, T + 1, 4) under `seen_actions` (K, T)."""
    trajectories = replay_all(programs, seen_actions)
    return (trajectories == seen_visible[None]).all(axis=(1, 2, 3))
