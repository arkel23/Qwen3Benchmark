I need a seeded stratified train/validation split of sample indices for our fine-grained datasets. numpy only.

Module-level function:

`stratified_split(labels, val_fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]`

- `labels` is a non-empty 1-D array-like of class labels (ints or strings). Index `i` refers to `labels[i]`.
- For each class with `n` samples, the number of validation samples is `floor(n * val_fraction)`, raised to 1 if that gives 0 and `n >= 2`. A class with a single sample puts it in train.
- Which samples of a class go to validation is chosen at random with a generator seeded by `seed`; the same `labels`, `val_fraction` and `seed` must always give the same split.
- Return `(train_idx, val_idx)`: two sorted `np.int64` arrays with no overlap whose union is all indices `0..len(labels)-1`.
- Raise `ValueError` if `labels` is empty or not 1-D, or if `val_fraction` is not strictly between 0 and 1.

Reply with a single Python code block containing the complete module.
