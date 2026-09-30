import numpy as np
import pytest
import torch

import solution


def _logits():
    return torch.tensor([
        [0.9, 0.5, 0.1, 0.0, -1.0],
        [0.1, 0.8, 0.7, 0.2, 0.0],
        [0.3, 0.2, 0.1, 0.9, 0.4],
        [0.0, 0.1, 0.2, 0.3, 0.4],
    ])


def test_top1():
    targets = torch.tensor([0, 1, 3, 0])
    assert solution.topk_accuracy(_logits(), targets, ks=(1,))[1] == pytest.approx(0.75, abs=1e-6)


def test_several_ks():
    targets = torch.tensor([1, 2, 4, 2])
    acc = solution.topk_accuracy(_logits(), targets, ks=(1, 2, 3))
    assert acc[1] == pytest.approx(0.0, abs=1e-6)
    assert acc[2] == pytest.approx(0.75, abs=1e-6)
    assert acc[3] == pytest.approx(1.0, abs=1e-6)


def test_k_equal_num_classes_is_one():
    targets = torch.tensor([4, 3, 2, 0])
    assert solution.topk_accuracy(_logits(), targets, ks=(5,))[5] == pytest.approx(1.0, abs=1e-6)


def test_returns_python_floats_keyed_by_k():
    acc = solution.topk_accuracy(_logits(), torch.tensor([0, 1, 3, 0]), ks=(3, 1))
    assert set(acc) == {1, 3}
    assert all(type(v) is float for v in acc.values())


def test_matches_numpy_on_random_data():
    torch.manual_seed(0)
    logits = torch.randn(200, 10)
    targets = torch.randint(0, 10, (200,))
    acc = solution.topk_accuracy(logits, targets, ks=(1, 3, 5))
    order = np.argsort(-logits.numpy(), axis=1)
    for k in (1, 3, 5):
        expected = (order[:, :k] == targets.numpy()[:, None]).any(axis=1).mean()
        assert acc[k] == pytest.approx(expected, abs=1e-6)


def test_requires_grad_input():
    logits = _logits().requires_grad_()
    acc = solution.topk_accuracy(logits, torch.tensor([0, 1, 3, 0]), ks=(1,))
    assert type(acc[1]) is float


def test_k_out_of_range_raises():
    with pytest.raises(ValueError):
        solution.topk_accuracy(_logits(), torch.tensor([0, 1, 3, 0]), ks=(6,))
    with pytest.raises(ValueError):
        solution.topk_accuracy(_logits(), torch.tensor([0, 1, 3, 0]), ks=(0,))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        solution.topk_accuracy(_logits(), torch.tensor([0, 1, 3]), ks=(1,))
    with pytest.raises(ValueError):
        solution.topk_accuracy(_logits()[0], torch.tensor([0]), ks=(1,))
