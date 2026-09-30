import copy

import pytest
import torch
from torch import nn

import solution


def _setup():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)).double()
    x = torch.randn(12, 4, dtype=torch.float64)
    y = torch.randn(12, 2, dtype=torch.float64)
    micro = [(x[i:i + 3], y[i:i + 3]) for i in range(0, 12, 3)]
    return model, x, y, micro


def _full_batch_step(model, x, y, lr=0.1):
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=lr)
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(reference(x), y)
    loss.backward()
    optimizer.step()
    return reference, loss.item()


def test_gradients_match_full_batch():
    model, x, y, micro = _setup()
    reference, _ = _full_batch_step(model, x, y)
    solution.accumulation_step(model, torch.optim.SGD(model.parameters(), lr=0.1), micro, nn.functional.mse_loss)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-12)


def test_parameters_match_full_batch():
    model, x, y, micro = _setup()
    reference, _ = _full_batch_step(model, x, y)
    solution.accumulation_step(model, torch.optim.SGD(model.parameters(), lr=0.1), micro, nn.functional.mse_loss)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-10, atol=1e-12)


def test_returns_mean_loss_float():
    model, x, y, micro = _setup()
    _, full_loss = _full_batch_step(model, x, y)
    loss = solution.accumulation_step(model, torch.optim.SGD(model.parameters(), lr=0.1), micro, nn.functional.mse_loss)
    assert type(loss) is float
    assert loss == pytest.approx(full_loss, rel=1e-10)


def test_optimizer_steps_once():
    model, _, _, micro = _setup()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = []
    original_step = optimizer.step
    optimizer.step = lambda *args, **kwargs: (calls.append(1), original_step(*args, **kwargs))[1]
    solution.accumulation_step(model, optimizer, micro, nn.functional.mse_loss)
    assert len(calls) == 1


def test_stale_gradients_cleared():
    model, x, y, micro = _setup()
    reference, _ = _full_batch_step(model, x, y)
    for p in model.parameters():
        p.grad = torch.full_like(p, 100.0)
    solution.accumulation_step(model, torch.optim.SGD(model.parameters(), lr=0.1), micro, nn.functional.mse_loss)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-10, atol=1e-12)


def test_adam_matches_full_batch():
    model, x, y, micro = _setup()
    reference = copy.deepcopy(model)
    ref_opt = torch.optim.Adam(reference.parameters(), lr=0.01)
    ref_opt.zero_grad()
    nn.functional.mse_loss(reference(x), y).backward()
    ref_opt.step()
    solution.accumulation_step(model, torch.optim.Adam(model.parameters(), lr=0.01), micro, nn.functional.mse_loss)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-8, atol=1e-10)


def test_empty_micro_batches_raise():
    model, _, _, _ = _setup()
    with pytest.raises(ValueError):
        solution.accumulation_step(model, torch.optim.SGD(model.parameters(), lr=0.1), [], nn.functional.mse_loss)
