import numpy as np
import pytest

import solution


def test_weight_zero_is_identity():
    values = np.random.default_rng(0).normal(size=20)
    np.testing.assert_allclose(solution.ema_smooth(values, 0.0), values, rtol=0, atol=1e-12)


def test_constant_sequence_unchanged():
    np.testing.assert_allclose(solution.ema_smooth([3.0] * 10, 0.9), np.full(10, 3.0), rtol=1e-12)


def test_hand_computed_values():
    np.testing.assert_allclose(solution.ema_smooth([1.0, 2.0, 3.0], 0.5), [1.0, 1.25 / 0.75, 2.125 / 0.875], rtol=1e-12)


def test_nan_positions_propagate_and_are_skipped():
    values = [1.0, np.nan, 2.0, np.nan, 3.0]
    out = solution.ema_smooth(values, 0.5)
    assert np.isnan(out[[1, 3]]).all()
    np.testing.assert_allclose(out[[0, 2, 4]], solution.ema_smooth([1.0, 2.0, 3.0], 0.5), rtol=1e-12)


def test_leading_nan():
    out = solution.ema_smooth([np.nan, np.nan, 4.0, 4.0], 0.8)
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [4.0, 4.0], rtol=1e-12)


def test_list_input_returns_float_array():
    out = solution.ema_smooth([1, 2, 3, 4], 0.6)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (4,)


def test_empty_input():
    out = solution.ema_smooth([], 0.5)
    assert out.shape == (0,)


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        solution.ema_smooth([1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        solution.ema_smooth([1.0, 2.0], -0.1)


def test_2d_input_raises():
    with pytest.raises(ValueError):
        solution.ema_smooth(np.ones((3, 2)), 0.5)
