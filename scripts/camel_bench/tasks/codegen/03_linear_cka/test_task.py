import numpy as np
import pytest

import solution


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(50, 8))
    Y = X @ rng.normal(size=(8, 5)) + 0.5 * rng.normal(size=(50, 5))
    return X, Y


def _hsic_cka(X, Y):
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K, L = X @ X.T, Y @ Y.T
    hsic = lambda A, B: np.trace(A @ H @ B @ H)
    return hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L))


def test_identical_inputs_give_one():
    X, _ = _data()
    assert solution.linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)


def test_matches_hsic_definition():
    X, Y = _data(1)
    X = X + np.linspace(-5.0, 5.0, 8)
    np.testing.assert_allclose(solution.linear_cka(X, Y), _hsic_cka(X, Y), rtol=1e-8)


def test_orthogonal_invariance():
    X, Y = _data(2)
    Q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(8, 8)))
    np.testing.assert_allclose(solution.linear_cka(X @ Q, Y), solution.linear_cka(X, Y), rtol=1e-8)


def test_isotropic_scaling_invariance():
    X, Y = _data(4)
    np.testing.assert_allclose(solution.linear_cka(7.5 * X, Y), solution.linear_cka(X, Y), rtol=1e-8)


def test_offset_invariance():
    X, Y = _data(5)
    offset = np.arange(8, dtype=float) * 10
    np.testing.assert_allclose(solution.linear_cka(X + offset, Y), solution.linear_cka(X, Y), rtol=1e-8)


def test_symmetric_bounded_float():
    X, Y = _data(6)
    value = solution.linear_cka(X, Y)
    assert type(value) is float
    assert 0.0 <= value <= 1.0
    np.testing.assert_allclose(value, solution.linear_cka(Y, X), rtol=1e-10)


def test_row_mismatch_raises():
    X, Y = _data()
    with pytest.raises(ValueError):
        solution.linear_cka(X, Y[:-1])


def test_non_2d_raises():
    with pytest.raises(ValueError):
        solution.linear_cka(np.arange(5.0), np.arange(5.0))


def test_constant_features_raise():
    X, _ = _data()
    with pytest.raises(ValueError):
        solution.linear_cka(X, np.ones((50, 3)))
