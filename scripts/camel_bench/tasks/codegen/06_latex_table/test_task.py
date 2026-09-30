import pandas as pd
import pytest

import solution


def _body_rows(tex):
    lines = tex.split("\n")
    return lines[4:lines.index("\\bottomrule")]


def test_prompt_example_exact():
    df = pd.DataFrame({"Model": ["a_b", "c"], "WER": [1.0, 2.5]})
    expected = "\n".join([
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Model & WER \\\\",
        "\\midrule",
        "a\\_b & \\textbf{1.00} \\\\",
        "c & 2.50 \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ])
    assert solution.df_to_latex(df, best={"WER": "min"}) == expected


def test_column_spec_mixed_types():
    df = pd.DataFrame({"name": ["x"], "n": [3], "acc": [0.5], "ok": [True]})
    assert solution.df_to_latex(df).split("\n")[0] == "\\begin{tabular}{lrrl}"


def test_escaping_header_and_cells():
    df = pd.DataFrame({"Acc %": [1.0], "run_id": ["#1 & $x$ {y}"]})
    lines = solution.df_to_latex(df).split("\n")
    assert lines[2] == "Acc \\% & run\\_id \\\\"
    assert lines[4] == "1.00 & \\#1 \\& \\$x\\$ \\{y\\} \\\\"


def test_bold_max():
    df = pd.DataFrame({"m": ["a", "b", "c"], "acc": [70.0, 81.25, 80.0]})
    assert _body_rows(solution.df_to_latex(df, best={"acc": "max"})) == [
        "a & 70.00 \\\\", "b & \\textbf{81.25} \\\\", "c & 80.00 \\\\"]


def test_bold_min_per_column():
    df = pd.DataFrame({"wer": [5.0, 3.0], "acc": [90.0, 80.0]})
    assert _body_rows(solution.df_to_latex(df, best={"wer": "min", "acc": "max"})) == [
        "5.00 & \\textbf{90.00} \\\\", "\\textbf{3.00} & 80.00 \\\\"]


def test_ties_all_bold():
    df = pd.DataFrame({"acc": [0.9, 0.5, 0.9]})
    assert _body_rows(solution.df_to_latex(df, best={"acc": "max"})) == [
        "\\textbf{0.90} \\\\", "0.50 \\\\", "\\textbf{0.90} \\\\"]


def test_nan_rendered_and_ignored():
    df = pd.DataFrame({"acc": [float("nan"), 0.4, 0.7]})
    assert _body_rows(solution.df_to_latex(df, best={"acc": "max"})) == [
        "-- \\\\", "0.40 \\\\", "\\textbf{0.70} \\\\"]


def test_float_fmt():
    df = pd.DataFrame({"acc": [0.123456]})
    assert _body_rows(solution.df_to_latex(df, float_fmt="{:.3f}")) == ["0.123 \\\\"]


def test_invalid_best_raises():
    df = pd.DataFrame({"m": ["a"], "acc": [1.0]})
    with pytest.raises(ValueError):
        solution.df_to_latex(df, best={"m": "max"})
    with pytest.raises(ValueError):
        solution.df_to_latex(df, best={"acc": "best"})
    with pytest.raises(ValueError):
        solution.df_to_latex(df, best={"missing": "max"})
