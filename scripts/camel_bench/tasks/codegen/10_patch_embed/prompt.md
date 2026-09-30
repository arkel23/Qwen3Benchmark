I need a ViT-style patch embedding layer in PyTorch.

Module-level class:

`class PatchEmbed(nn.Module)` with
`__init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, use_cls_token: bool = True)`

Attributes:
- `self.proj`: `nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)` (with bias).
- `self.num_patches`: `(img_size // patch_size) ** 2`.
- `self.cls_token`: an `nn.Parameter` of shape `(1, 1, embed_dim)` when `use_cls_token` is True, otherwise `None`.
- `self.pos_embed`: an `nn.Parameter` of shape `(1, num_patches + 1, embed_dim)` with a CLS token, or `(1, num_patches, embed_dim)` without.
- Initialise `cls_token` and `pos_embed` with `nn.init.trunc_normal_(..., std=0.02)`.
- Raise `ValueError` in `__init__` if `img_size` is not divisible by `patch_size`.

`forward(self, x: torch.Tensor) -> torch.Tensor`:
- `x` has shape `(B, in_chans, img_size, img_size)`; raise `ValueError` if its height or width differs from `img_size`.
- Apply `proj`, then turn the `(B, D, H', W')` grid into a token sequence `(B, H'*W', D)` in row-major order: token `r * W' + c` is grid position row `r`, column `c`.
- If there is a CLS token, prepend it (expanded over the batch) at index 0.
- Add `pos_embed` and return a tensor of shape `(B, num_patches + 1, embed_dim)` with a CLS token or `(B, num_patches, embed_dim)` without.

Reply with a single Python code block containing the complete module.
