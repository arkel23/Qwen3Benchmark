I keep hand-editing LaTeX tables from pandas results. Please write a function that turns a DataFrame into a booktabs `tabular` string. pandas only (do not use `DataFrame.to_latex`).

Module-level function:

`df_to_latex(df: pd.DataFrame, best: dict[str, str] | None = None, float_fmt: str = "{:.2f}") -> str`

Output format, lines joined with `"\n"` and no trailing newline:

```
\begin{tabular}{<colspec>}
\toprule
<header cells joined by " & "> \\
\midrule
<one line per row: cells joined by " & "> \\
\bottomrule
\end{tabular}
```

Rules:
- A column is numeric if `pd.api.types.is_numeric_dtype` is true for it and it is not a bool column. `<colspec>` has one letter per column in order: `r` for numeric columns, `l` for all others.
- Header cells are the column names converted with `str`.
- Numeric cells are formatted with `float_fmt.format(value)`; a missing value (NaN) is written as `--`.
- Non-numeric cells are the value converted with `str`.
- Header cells and non-numeric cells are LaTeX-escaped: each of `% & _ # $ { }` gets a backslash in front (`a_b` -> `a\_b`). Nothing else is escaped.
- `best` maps a numeric column name to `"max"` or `"min"`. In that column, every cell whose value equals the column's max (or min), ignoring NaN, is wrapped as `\textbf{<formatted value>}`; ties are all bolded.
- Raise `ValueError` if a key of `best` is not a numeric column of `df`, or a value is not `"max"` or `"min"`.

Example: `pd.DataFrame({"Model": ["a_b", "c"], "WER": [1.0, 2.5]})` with `best={"WER": "min"}` gives

```
\begin{tabular}{lr}
\toprule
Model & WER \\
\midrule
a\_b & \textbf{1.00} \\
c & 2.50 \\
\bottomrule
\end{tabular}
```

Reply with a single Python code block containing the complete module.
