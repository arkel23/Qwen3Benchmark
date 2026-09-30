import pytest

import solution


def test_levenshtein_classic():
    assert solution.levenshtein("kitten", "sitting") == 3


def test_levenshtein_empty():
    assert solution.levenshtein("", "abc") == 3
    assert solution.levenshtein("abc", "") == 3
    assert solution.levenshtein("", "") == 0


def test_normalize_default_unchanged():
    assert solution.normalize("  Hi, there!  ") == "  Hi, there!  "


def test_normalize_strip_punctuation():
    assert solution.normalize("a, b!", strip_punctuation=True) == "a b"


def test_normalize_collapse_whitespace():
    assert solution.normalize("  a \t b\n", collapse_whitespace=True) == "a b"


def test_normalize_punctuation_before_whitespace():
    assert solution.normalize("a , b", strip_punctuation=True, collapse_whitespace=True) == "a b"


def test_cer_is_corpus_level():
    assert solution.cer(["abcd", "ab"], ["abcd", "xy"]) == pytest.approx(2 / 6, abs=1e-12)


def test_cer_can_exceed_one():
    assert solution.cer(["a"], ["bcd"]) == pytest.approx(3.0, abs=1e-12)


def test_cer_applies_flags():
    assert solution.cer(["Hello, world"], ["hello world"], strip_punctuation=True) == pytest.approx(1 / 11, abs=1e-12)


def test_cer_errors():
    with pytest.raises(ValueError):
        solution.cer(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        solution.cer([""], ["a"])
    with pytest.raises(ValueError):
        solution.cer(["!!"], ["a"], strip_punctuation=True)
