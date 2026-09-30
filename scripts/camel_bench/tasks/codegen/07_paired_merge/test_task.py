import pandas as pd
import pytest

import solution


def _frames():
    left = pd.DataFrame({"lang": ["fr", "de", "en"], "wer": [10.0, 12.0, 5.0], "note": ["x", "y", "z"]})
    right = pd.DataFrame({"lang": ["en", "fr", "de"], "wer": [4.0, 11.5, 12.0], "extra": [1, 2, 3]})
    return left, right


def test_difference_values_sorted():
    left, right = _frames()
    out = solution.paired_difference(left, right, ["lang"], "wer")
    assert out["lang"].tolist() == ["de", "en", "fr"]
    assert out["diff"].tolist() == pytest.approx([0.0, 1.0, -1.5], abs=1e-12)


def test_columns_and_index():
    left, right = _frames()
    out = solution.paired_difference(left, right, ["lang"], "wer")
    assert list(out.columns) == ["lang", "wer_left", "wer_right", "diff"]
    assert out.index.tolist() == [0, 1, 2]
    assert out["wer_right"].tolist() == pytest.approx([12.0, 4.0, 11.5], abs=1e-12)


def test_multiple_keys():
    left = pd.DataFrame({"model": ["b", "a", "a"], "lang": ["en", "fr", "en"], "acc": [1.0, 2.0, 3.0]})
    right = pd.DataFrame({"model": ["a", "a", "b"], "lang": ["en", "fr", "en"], "acc": [0.5, 2.0, 0.0]})
    out = solution.paired_difference(left, right, ["model", "lang"], "acc")
    assert list(zip(out["model"], out["lang"])) == [("a", "en"), ("a", "fr"), ("b", "en")]
    assert out["diff"].tolist() == pytest.approx([2.5, 0.0, 1.0], abs=1e-12)


def test_inputs_not_modified():
    left, right = _frames()
    left_copy, right_copy = left.copy(), right.copy()
    solution.paired_difference(left, right, ["lang"], "wer")
    pd.testing.assert_frame_equal(left, left_copy)
    pd.testing.assert_frame_equal(right, right_copy)


def test_duplicate_left_raises():
    left, right = _frames()
    left = pd.concat([left, left.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        solution.paired_difference(left, right, ["lang"], "wer")


def test_duplicate_right_raises():
    left, right = _frames()
    right = pd.concat([right, right.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        solution.paired_difference(left, right, ["lang"], "wer")


def test_key_only_in_right_raises():
    left, right = _frames()
    right = pd.concat([right, pd.DataFrame({"lang": ["ja"], "wer": [9.0], "extra": [4]})], ignore_index=True)
    with pytest.raises(ValueError):
        solution.paired_difference(left, right, ["lang"], "wer")


def test_key_only_in_left_raises():
    left, right = _frames()
    with pytest.raises(ValueError):
        solution.paired_difference(left, right.iloc[:2], ["lang"], "wer")


def test_missing_column_raises():
    left, right = _frames()
    with pytest.raises(ValueError):
        solution.paired_difference(left, right.drop(columns="wer"), ["lang"], "wer")
