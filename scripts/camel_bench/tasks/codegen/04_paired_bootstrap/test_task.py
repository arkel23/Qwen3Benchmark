import numpy as np
import pytest

import solution


def _scores(seed=0, n=40):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    return base + 0.3 + 0.1 * rng.normal(size=n), base


def test_matches_specified_procedure():
    a, b = _scores()
    mean, lower, upper = solution.paired_bootstrap_ci(a, b, n_boot=2000, alpha=0.1, seed=7)
    d = a - b
    idx = np.random.default_rng(7).integers(0, d.size, size=(2000, d.size))
    boot = d[idx].mean(axis=1)
    np.testing.assert_allclose([mean, lower, upper], [d.mean(), np.percentile(boot, 5), np.percentile(boot, 95)], rtol=1e-10)


def test_deterministic_given_seed():
    a, b = _scores(1)
    assert solution.paired_bootstrap_ci(a, b, n_boot=500, seed=3) == solution.paired_bootstrap_ci(a, b, n_boot=500, seed=3)


def test_different_seeds_differ():
    a, b = _scores(2)
    first = solution.paired_bootstrap_ci(a, b, n_boot=500, seed=0)
    second = solution.paired_bootstrap_ci(a, b, n_boot=500, seed=1)
    assert first[1:] != second[1:]


def test_returns_floats_ordered():
    a, b = _scores(3)
    mean, lower, upper = solution.paired_bootstrap_ci(list(a), list(b), n_boot=500)
    assert all(type(v) is float for v in (mean, lower, upper))
    assert lower <= mean <= upper


def test_constant_difference_is_degenerate():
    a = np.arange(10.0)
    mean, lower, upper = solution.paired_bootstrap_ci(a + 2.0, a, n_boot=300)
    np.testing.assert_allclose([mean, lower, upper], [2.0, 2.0, 2.0], atol=1e-12)


def test_paired_interval_is_narrow_for_correlated_scores():
    a, b = _scores(4, n=60)
    _, lower, upper = solution.paired_bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert upper - lower < 0.1


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        solution.paired_bootstrap_ci([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        solution.paired_bootstrap_ci([1.0], [2.0])
    with pytest.raises(ValueError):
        solution.paired_bootstrap_ci(np.ones((3, 2)), np.ones((3, 2)))


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        solution.paired_bootstrap_ci([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], alpha=0.0)
    with pytest.raises(ValueError):
        solution.paired_bootstrap_ci([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], alpha=1.5)
