import pandas as pd


def paired_difference(left: pd.DataFrame, right: pd.DataFrame, keys, metric: str) -> pd.DataFrame:
    keys = list(keys)
    columns = keys + [metric]
    for side, df in (("left", left), ("right", right)):
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"{side} lacks columns {sorted(missing)}")
    merged = left[columns].merge(right[columns], on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    merged["diff"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    return merged.sort_values(keys).reset_index(drop=True)
