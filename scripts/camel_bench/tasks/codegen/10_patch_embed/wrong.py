import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, use_cls_token=True):
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"img_size {img_size} is not divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + int(use_cls_token), embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"expected {self.img_size}x{self.img_size} input, got {tuple(x.shape[-2:])}")
        x = self.proj(x).permute(0, 3, 2, 1).flatten(1, 2)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self.pos_embed
