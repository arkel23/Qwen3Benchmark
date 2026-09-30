import numpy as np


def paired_bootstrap_ci(a, b, n_boot=10000, alpha=0.05, seed=0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError(f"expected two 1-D arrays of equal length >= 2, got {a.shape} and {b.shape}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    n = a.size
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, n, size=(n_boot, n))
    idx_b = rng.integers(0, n, size=(n_boot, n))
    boot_means = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float((a - b).mean()), float(lower), float(upper)
