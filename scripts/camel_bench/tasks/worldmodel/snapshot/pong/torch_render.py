"""GPU rendering of visible states, identical to pong.render.frames."""
import torch

from pong.render import BALL, PADDLE, PIP, tiles


class Renderer:
    def __init__(self, cell, device):
        self.cell = cell
        self.tiles = torch.from_numpy(tiles(cell)).to(device)
        self.rows = torch.arange(16, device=device)

    def __call__(self, visible, heights):
        """visible (..., T, 4) int64 and heights (...,) -> uint8 frames (..., T, 3, 16 * cell, 16 * cell)."""
        top, x, y, misses = visible.unbind(-1)
        lead = visible.shape[:-1]
        grid = torch.zeros(*lead, 16, 16, dtype=torch.long, device=visible.device)
        bottom = top + heights.reshape(*heights.shape, *([1] * (top.dim() - heights.dim())))
        grid[..., 0] = torch.where((self.rows >= top[..., None]) & (self.rows < bottom[..., None]), PADDLE, 0)
        grid[..., 0, :] = torch.where(((misses[..., None] >> self.rows) & 1) == 1, PIP, 0)
        flat = grid.reshape(-1, 16, 16)
        idx = torch.arange(flat.shape[0], device=visible.device)
        flat[idx, y.reshape(-1), x.reshape(-1)] = BALL
        image = self.tiles[flat]
        c = self.cell
        image = image.permute(0, 1, 3, 2, 4, 5).reshape(-1, 16 * c, 16 * c, 3)
        return image.permute(0, 3, 1, 2).reshape(*lead, 3, 16 * c, 16 * c)
