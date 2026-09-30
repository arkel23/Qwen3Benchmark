import numpy as np


def ema_smooth(values, weight: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D sequence, got shape {values.shape}")
    if not 0 <= weight < 1:
        raise ValueError(f"weight must lie in [0, 1), got {weight}")
    smoothed = np.full(values.shape, np.nan)
    last = 0.0
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        last = last * weight + (1 - weight) * v
        smoothed[i] = last / (1 - weight ** (i + 1))
    return smoothed
