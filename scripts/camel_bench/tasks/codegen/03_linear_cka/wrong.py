import numpy as np


def linear_cka(X, Y) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"expected 2-D arrays with equal rows, got {X.shape} and {Y.shape}")
    X = X - X.mean()
    Y = Y - Y.mean()
    if not X.any() or not Y.any():
        raise ValueError("centred features are all zero")
    cross = np.linalg.norm(Y.T @ X) ** 2
    return float(cross / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))
