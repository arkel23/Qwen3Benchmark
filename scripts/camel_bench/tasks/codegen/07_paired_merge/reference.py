import pandas as pd


def paired_difference(left: pd.DataFrame, right: pd.DataFrame, keys, metric: str) -> pd.DataFrame:
    keys = list(keys)
    columns = keys + [metric]
    for side, df in (("left", left), ("right", right)):
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"{side} lacks columns {sorted(missing)}")
        if df.duplicated(keys).any():
            raise ValueError(f"{side} has duplicate keys")
    merged = left[columns].merge(right[columns], on=keys, how="outer",
                                 suffixes=("_left", "_right"), indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("keys do not match one-to-one")
    merged["diff"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    return merged.drop(columns="_merge").sort_values(keys).reset_index(drop=True)
