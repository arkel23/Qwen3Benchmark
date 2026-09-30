import pytest
import torch

import solution


def _module(**kwargs):
    torch.manual_seed(0)
    params = dict(img_size=32, patch_size=8, in_chans=3, embed_dim=16)
    params.update(kwargs)
    return solution.PatchEmbed(**params)


def test_output_shape_with_cls():
    out = _module()(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 17, 16)


def test_output_shape_without_cls():
    module = _module(use_cls_token=False)
    assert module.cls_token is None
    assert module(torch.randn(2, 3, 32, 32)).shape == (2, 16, 16)


def test_attributes_and_parameter_shapes():
    module = _module(img_size=24, patch_size=4, in_chans=1, embed_dim=10)
    assert module.num_patches == 36
    assert module.proj.kernel_size == (4, 4) and module.proj.stride == (4, 4)
    assert isinstance(module.cls_token, torch.nn.Parameter) and module.cls_token.shape == (1, 1, 10)
    assert isinstance(module.pos_embed, torch.nn.Parameter) and module.pos_embed.shape == (1, 37, 10)


def test_indivisible_img_size_raises():
    with pytest.raises(ValueError):
        solution.PatchEmbed(img_size=30, patch_size=8)


def test_wrong_input_size_raises():
    with pytest.raises(ValueError):
        _module()(torch.randn(1, 3, 24, 32))


def test_row_major_patch_order():
    module = _module(use_cls_token=False)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        module.pos_embed.zero_()
        grid = module.proj(x)
        out = module(x)
    for r in range(4):
        for c in range(4):
            torch.testing.assert_close(out[:, r * 4 + c], grid[:, :, r, c], rtol=1e-5, atol=1e-6)


def test_cls_token_first_with_position():
    module = _module()
    with torch.no_grad():
        out = module(torch.randn(3, 3, 32, 32))
    expected = (module.cls_token[0, 0] + module.pos_embed[0, 0]).expand(3, -1)
    torch.testing.assert_close(out[:, 0], expected, rtol=1e-5, atol=1e-6)


def test_positional_embedding_added():
    module = _module()
    with torch.no_grad():
        module.proj.weight.zero_()
        module.proj.bias.zero_()
        module.cls_token.zero_()
        out = module(torch.randn(2, 3, 32, 32))
    torch.testing.assert_close(out, module.pos_embed.detach().expand(2, -1, -1), rtol=0, atol=1e-7)


def test_gradients_reach_embeddings():
    module = _module()
    module(torch.randn(2, 3, 32, 32)).sum().backward()
    assert module.pos_embed.grad is not None and module.cls_token.grad is not None
    assert module.proj.weight.grad is not None
