import numpy as np


def stratified_split(labels, val_fraction: float, seed: int = 0):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"expected a non-empty 1-D label array, got shape {labels.shape}")
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == label))
        n_val = max(int(round(idx.size * val_fraction)), 1) if idx.size >= 2 else 0
        val.append(idx[:n_val])
        train.append(idx[n_val:])
    return np.sort(np.concatenate(train)).astype(np.int64), np.sort(np.concatenate(val)).astype(np.int64)
