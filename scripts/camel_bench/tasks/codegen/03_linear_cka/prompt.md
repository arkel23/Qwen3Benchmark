I want a numpy implementation of linear CKA (centered kernel alignment, Kornblith et al. 2019) to compare representations of two networks on the same inputs.

Module-level function:

`linear_cka(X: np.ndarray, Y: np.ndarray) -> float`

- `X` has shape `(n, p1)` and `Y` has shape `(n, p2)`: rows are the same n examples, columns are features. `p1` and `p2` may differ. Accept array-likes and work in float64.
- Centre each column of `X` and `Y` (subtract that column's mean), then return
  `||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)`
  as a Python `float`. This equals the HSIC-based definition with linear kernels and the centring matrix `H = I - 11^T/n`.
- The result is 1 for identical inputs, symmetric in `X` and `Y`, in `[0, 1]`, and invariant to orthogonal transforms, isotropic scaling and adding a per-feature constant offset to either input.
- Raise `ValueError` if either input is not 2-D, if the numbers of rows differ, or if either centred matrix is entirely zero (e.g. constant features).

Reply with a single Python code block containing the complete module.
