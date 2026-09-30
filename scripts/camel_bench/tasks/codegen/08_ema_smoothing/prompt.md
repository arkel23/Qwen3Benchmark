I want to smooth training curves the way TensorBoard's smoothing slider does: an exponential moving average with bias correction, robust to NaN entries in logged metrics. numpy only.

Module-level function:

`ema_smooth(values, weight: float) -> np.ndarray`

- `values` is a 1-D array-like; convert to float64. `weight` is the smoothing factor in `[0, 1)`.
- Keep a running state `last = 0.0` and a counter `n = 0` of non-NaN values seen so far. For each non-NaN value `v`, in order:
  ```
  last = last * weight + (1 - weight) * v
  n += 1
  smoothed = last / (1 - weight ** n)
  ```
- NaN entries do not update `last` or `n`; the output at their positions is NaN.
- Return a float64 array of the same length as the input. `weight=0` returns the input unchanged; an empty input returns an empty array; a constant sequence stays constant.
- Raise `ValueError` if `values` is not 1-D or `weight` is outside `[0, 1)`.

Reply with a single Python code block containing the complete module.
