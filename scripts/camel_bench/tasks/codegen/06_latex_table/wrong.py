import pandas as pd

_ESCAPES = str.maketrans({c: "\\" + c for c in "%&_#${}"})


def escape_latex(text: str) -> str:
    return text.translate(_ESCAPES)


def df_to_latex(df: pd.DataFrame, best=None, float_fmt="{:.2f}") -> str:
    best = best or {}
    numeric = {c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])}
    for col, mode in best.items():
        if col not in numeric or mode not in ("max", "min"):
            raise ValueError(f"cannot pick {mode!r} of column {col!r}")
    best_rows = {col: getattr(df[col].reset_index(drop=True), "idx" + mode)() for col, mode in best.items()}

    def cell(col, i):
        value = df[col].iloc[i]
        if col not in numeric:
            return escape_latex(str(value))
        if pd.isna(value):
            return "--"
        text = float_fmt.format(value)
        return f"\\textbf{{{text}}}" if best_rows.get(col) == i else text

    lines = ["\\begin{tabular}{" + "".join("r" if c in numeric else "l" for c in df.columns) + "}",
             "\\toprule",
             " & ".join(escape_latex(str(c)) for c in df.columns) + " \\\\",
             "\\midrule"]
    for i in range(len(df)):
        lines.append(" & ".join(cell(c, i) for c in df.columns) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)
