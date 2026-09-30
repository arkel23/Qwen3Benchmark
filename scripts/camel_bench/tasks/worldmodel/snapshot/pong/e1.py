"""E1: a from-scratch model from rendered rollouts and actions to per-slot program predictions."""
import torch
from torch import nn

from pong.data import SLOT_SIZES


class FrameEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Flatten(), nn.LazyLinear(width),
        )

    def forward(self, frames):
        return self.net(frames)


class ProgramFromVideo(nn.Module):
    """frames (B, K, T + 1, 3, H, W) uint8 and actions (B, K, T) in {-1, 0, 1} -> list of slot logits."""

    def __init__(self, width=192, layers=4, heads=4, max_frames=512):
        super().__init__()
        self.frames = FrameEncoder(width)
        self.actions = nn.Embedding(4, width)
        self.position = nn.Parameter(torch.zeros(1, max_frames, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers)
        self.heads = nn.ModuleList(nn.Linear(width, size) for size in SLOT_SIZES)

    def forward(self, frames, actions):
        b, k, t = frames.shape[:3]
        x = self.frames(frames.reshape(b * k * t, *frames.shape[3:]).float() / 255).reshape(b * k, t, -1)
        previous = torch.cat([torch.full_like(actions[..., :1], 2), actions], dim=-1) + 1  # 3 = no action yet
        x = x + self.actions(previous.reshape(b * k, t)) + self.position[:, :t]
        x = self.temporal(x).mean(dim=1).reshape(b, k, -1).mean(dim=1)
        return [head(x) for head in self.heads]
