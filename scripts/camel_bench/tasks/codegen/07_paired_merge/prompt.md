I compare per-language (or per-dataset) metrics of two runs stored in two pandas DataFrames. I want a strict paired merge that refuses to silently drop or duplicate rows.

Module-level function:

`paired_difference(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame`

- `keys` are the columns identifying a row; `metric` is a numeric column present in both frames.
- Raise `ValueError` if any of `keys` or `metric` is missing from either frame.
- Raise `ValueError` if either frame has duplicate key combinations.
- Raise `ValueError` if the set of key combinations in `left` differs from that in `right` (a key present on only one side).
- Otherwise return a new DataFrame with exactly the columns `[*keys, f"{metric}_left", f"{metric}_right", "diff"]` in that order, where `diff = metric_left - metric_right`. Other columns of the inputs are not included.
- Rows are sorted ascending by `keys` (in the order given) and the index is a fresh `RangeIndex` starting at 0.
- Do not modify the input frames.

Reply with a single Python code block containing the complete module.
