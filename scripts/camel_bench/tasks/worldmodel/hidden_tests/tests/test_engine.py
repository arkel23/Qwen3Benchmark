import numpy as np
import pytest

from pong.classes import consistent_programs, equivalence_classes, probe_actions, replay_all
from pong.render import frames
from pong.rules import SLOTS, all_programs, compile_program
from pong.sim import simulate

PROGRAMS = all_programs()


def test_enumeration_count():
    assert len(PROGRAMS) == int(np.prod([len(v) for v in SLOTS.values()])) == 288
    assert len({p.index() for p in PROGRAMS}) == 288


def test_rollouts_are_deterministic():
    seeds = np.arange(len(PROGRAMS))
    first, actions = simulate(PROGRAMS, 128, seeds=seeds)
    second, again = simulate(PROGRAMS, 128, seeds=seeds)
    assert np.array_equal(first, second) and np.array_equal(actions, again)


def test_replaying_recorded_actions_reproduces_the_rollout():
    visible, actions = simulate(PROGRAMS, 128, seeds=np.arange(len(PROGRAMS)))
    replayed, _ = simulate(PROGRAMS, 128, actions=actions)
    assert np.array_equal(visible, replayed)


@pytest.mark.parametrize("index", range(0, 288, 7))
def test_compiled_python_matches_the_simulator(index):
    program = PROGRAMS[index]
    visible, actions = simulate([program], 256, seeds=[index])
    game = compile_program(program)()
    trajectory = [(game.top, game.x, game.y, game.misses)] + [game.step(int(a)) for a in actions[0]]
    assert np.array_equal(np.array(trajectory), visible[0])


def test_rules_are_exercised():
    visible, _ = simulate(PROGRAMS, 256, seeds=np.arange(len(PROGRAMS)))
    x, misses = visible[..., 1], visible[..., 3]
    hits = ((x[:, :-1] == 1) & (x[:, 1:] == 2)).sum(axis=1)
    assert (hits > 0).mean() > 0.9 and (misses[:, -1] > 0).mean() > 0.9


def test_rendering_is_injective_on_visible_state():
    visible, _ = simulate(PROGRAMS[:32], 64, seeds=np.arange(32))
    heights = np.array([p.paddle_height for p in PROGRAMS[:32]])
    image = frames(visible, heights, cell=4)
    assert image.shape == (32, 65, 64, 64, 3)
    keys = {}
    for i in range(32):
        for t in range(65):
            state = (heights[i], *visible[i, t])
            keys.setdefault(image[i, t].tobytes(), set()).add(state)
    assert all(len(states) == 1 for states in keys.values())


def test_classes_group_only_behaviourally_equal_programs():
    ids = equivalence_classes(PROGRAMS, probe_actions())
    assert ids.max() + 1 == 200  # measured by scripts/probe_sufficiency.py
    held_out = probe_actions(ticks=512, controlled=64, random=64, seed=1)
    trajectories = replay_all(PROGRAMS, held_out)
    for cls in np.unique(ids):
        members = np.flatnonzero(ids == cls)
        assert all(np.array_equal(trajectories[members[0]], trajectories[m]) for m in members[1:])


def test_true_program_is_always_consistent_with_its_own_rollouts():
    for index in (0, 101, 287):
        visible, actions = simulate([PROGRAMS[index]] * 4, 128, seeds=[1, 2, 3, 4])
        assert consistent_programs(PROGRAMS, actions, visible)[index]


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a GPU")
def test_gpu_renderer_matches_numpy_renderer():
    import torch
    from pong.torch_render import Renderer
    visible, _ = simulate(PROGRAMS[:8], 64, seeds=np.arange(8))
    heights = np.array([p.paddle_height for p in PROGRAMS[:8]])
    expected = frames(visible, heights, cell=4).transpose(0, 1, 4, 2, 3)
    rendered = Renderer(4, "cuda")(torch.from_numpy(visible).cuda(), torch.from_numpy(heights).cuda())
    assert np.array_equal(rendered.cpu().numpy(), expected)
