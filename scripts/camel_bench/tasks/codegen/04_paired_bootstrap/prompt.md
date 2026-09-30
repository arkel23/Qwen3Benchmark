I compare two systems evaluated on the same test items and want a paired percentile bootstrap confidence interval for the mean difference. numpy only.

Module-level function:

`paired_bootstrap_ci(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]`

- `a` and `b` are 1-D array-likes of per-item scores for the two systems; item `i` of `a` is paired with item `i` of `b`. Convert to float64.
- Let `d = a - b`. Resample the items (pairs), not `a` and `b` independently, using exactly this procedure so results are reproducible:
  ```
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, n, size=(n_boot, n))
  boot_means = d[idx].mean(axis=1)
  ```
- `lower` and `upper` are `np.percentile(boot_means, 100 * alpha / 2)` and `np.percentile(boot_means, 100 * (1 - alpha / 2))` with numpy's default (linear) interpolation.
- Return `(d.mean(), lower, upper)` as three Python `float`s. The same inputs and seed must give identical output.
- Raise `ValueError` if `a` or `b` is not 1-D, if their lengths differ, if there are fewer than 2 items, or if `alpha` is not strictly between 0 and 1.

Reply with a single Python code block containing the complete module.
