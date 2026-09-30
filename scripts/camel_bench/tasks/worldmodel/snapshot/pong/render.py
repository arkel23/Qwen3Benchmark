"""Render visible states to RGB frames. Misses are drawn in binary on the score row, so rendering is injective."""
import numpy as np

BACKGROUND, PADDLE, BALL, PIP = 0, 1, 2, 3
COLOURS = np.array([(18, 20, 28), (80, 200, 255), (255, 214, 64), (220, 90, 90)], dtype=np.uint8)


def tiles(cell):
    """Tile bitmaps (4, cell, cell, 3): background, striped paddle, disc ball, square score pip."""
    u, v = np.meshgrid(np.arange(cell), np.arange(cell), indexing="ij")
    centre = (cell - 1) / 2
    masks = np.stack([
        np.zeros((cell, cell), dtype=bool),
        (v >= cell // 4) & (v < cell - cell // 4) & ((u // max(cell // 4, 1)) % 2 == 0),
        (u - centre) ** 2 + (v - centre) ** 2 <= (cell * 0.4) ** 2,
        (abs(u - centre) <= cell * 0.3) & (abs(v - centre) <= cell * 0.3),
    ])
    out = np.broadcast_to(COLOURS[BACKGROUND], (4, cell, cell, 3)).copy()
    for kind in (PADDLE, BALL, PIP):
        out[kind][masks[kind]] = COLOURS[kind]
    return out


def frames(visible, heights, cell):
    """visible (N, T, 4) and paddle heights (N,) -> uint8 frames (N, T, 16 * cell, 16 * cell, 3)."""
    n, t = visible.shape[:2]
    top, x, y, misses = (visible[..., i] for i in range(4))
    rows = np.arange(16)
    grid = np.zeros((n, t, 16, 16), dtype=np.int64)
    in_paddle = (rows >= top[..., None]) & (rows < (top + heights[:, None])[..., None])
    grid[..., 0] = np.where(in_paddle, PADDLE, BACKGROUND)
    bits = (misses[..., None] >> rows) & 1
    grid[..., 0, :] = np.where(bits == 1, PIP, BACKGROUND)
    ni, ti = np.meshgrid(np.arange(n), np.arange(t), indexing="ij")
    grid[ni, ti, y, x] = BALL
    image = tiles(cell)[grid]
    return image.transpose(0, 1, 2, 4, 3, 5, 6).reshape(n, t, 16 * cell, 16 * cell, 3)
