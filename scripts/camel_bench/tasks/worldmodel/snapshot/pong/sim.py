"""Batched integer wall-ball simulator and scripted controller; the single source of truth for data and scoring.

Grid: 16 columns (paddle in column 0, ball in 1..15) and 16 rows (row 0 is the score bar, field rows 1..15).
Tick order: paddle action, ball phase, paddle hit, right wall, top/bottom wall, miss and reset.
Visible state per tick: (paddle top, ball x, ball y, misses).
"""
import numpy as np

from pong.rules import SLOTS

WALL_WRAP = SLOTS["wall"].index("wrap")
HIT = {name: i for i, name in enumerate(SLOTS["hit_vy"])}
MODES = ("aim_top", "aim_middle", "aim_bottom", "edge_up", "edge_down", "miss")
SEGMENT, RANDOM_P = 16, 0.2


def params(programs):
    """Slot values as integer columns (N,) from a list of Program."""
    return {
        "height": np.array([p.paddle_height for p in programs]),
        "speed": np.array([p.paddle_speed for p in programs]),
        "period": np.array([p.ball_period for p in programs]),
        "wrap": np.array([p.wall == "wrap" for p in programs]),
        "hit": np.array([HIT[p.hit_vy] for p in programs]),
        "serve": np.array([p.serve_vy for p in programs]),
    }


def controller_plan(seeds, ticks):
    """Per-rollout mode schedule and random overrides, fixed by the seed."""
    modes = np.empty((len(seeds), ticks), dtype=np.int64)
    overrides = np.empty((len(seeds), ticks), dtype=np.int64)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        modes[i] = np.repeat(rng.integers(len(MODES), size=-(-ticks // SEGMENT)), SEGMENT)[:ticks]
        random_action = rng.integers(-1, 2, size=ticks)
        overrides[i] = np.where(rng.random(ticks) < RANDOM_P, random_action, 2)
    return modes, overrides


def controller_action(mode, override, top, y, height):
    offset = np.select([mode == 0, mode == 1, mode == 2], [0, (height - 1) // 2, height - 1], 0)
    aim = np.sign(y - offset - top)
    away = np.where(y >= top, -1, 1)
    action = np.select([mode <= 2, mode == 3, mode == 4], [aim, -1, 1], away)
    return np.where(override == 2, action, override)


def simulate(programs, ticks, seeds=None, actions=None):
    """Closed-loop rollouts from `seeds`, or open-loop replay of `actions` (N, ticks).

    Returns visible states (N, ticks + 1, 4) and the actions taken (N, ticks), both int64.
    """
    p = params(programs)
    n = len(programs)
    top = 8 - p["height"] // 2
    x, y = np.full(n, 8), np.full(n, 8)
    vx, vy = np.full(n, -1), p["serve"].copy()
    phase, misses = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    if actions is None:
        modes, overrides = controller_plan(seeds, ticks)
        actions = np.empty((n, ticks), dtype=np.int64)
    visible = np.empty((n, ticks + 1, 4), dtype=np.int64)
    visible[:, 0] = np.stack([top, x, y, misses], axis=1)
    for t in range(ticks):
        if seeds is not None:
            actions[:, t] = controller_action(modes[:, t], overrides[:, t], top, y, p["height"])
        previous = top
        top = np.clip(top + actions[:, t] * p["speed"], 1, 16 - p["height"])
        last_move = np.sign(top - previous)
        phase = (phase + 1) % p["period"]
        moving = phase == 0

        hit = moving & (x == 1) & (vx == -1) & (y >= top) & (y < top + p["height"])
        segment = np.sign(2 * (y - top) - (p["height"] - 1))
        vy = np.where(hit, np.choose(p["hit"], [vy, -vy, segment, last_move]), vy)
        vx = np.where(hit, 1, vx)
        vx = np.where(moving & (x == 15) & (vx == 1), -1, vx)

        nx, ny = x + vx, y + vy
        out = (ny < 1) | (ny > 15)
        bounce = out & ~p["wrap"]
        vy = np.where(bounce, -vy, vy)
        ny = np.where(bounce, y + vy, ny)
        ny = np.where(out & p["wrap"], (ny - 1) % 15 + 1, ny)

        miss = moving & (nx == 0)
        misses = misses + miss
        x = np.where(miss, 8, np.where(moving, nx, x))
        y = np.where(miss, 8, np.where(moving, ny, y))
        vx = np.where(miss, -1, vx)
        vy = np.where(miss, p["serve"], vy)
        phase = np.where(miss, 0, phase)
        visible[:, t + 1] = np.stack([top, x, y, misses], axis=1)
    return visible, actions
