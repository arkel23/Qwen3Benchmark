import pytest

import solution


def test_half_rounds_up():
    assert solution.round_half_up(0.125, 2) == "0.13"


def test_binary_representation_ignored():
    assert solution.round_half_up(2.675, 2) == "2.68"
    assert solution.round_half_up(1.005, 2) == "1.01"


def test_zero_decimals():
    assert solution.round_half_up(2.5, 0) == "3"
    assert solution.round_half_up(0.5, 0) == "1"


def test_negative_half_away_from_zero():
    assert solution.round_half_up(-2.5, 0) == "-3"
    assert solution.round_half_up(-0.125, 2) == "-0.13"


def test_pads_trailing_zeros():
    assert solution.round_half_up(1.5, 3) == "1.500"


def test_default_decimals():
    assert solution.round_half_up(3.14159) == "3.14"


def test_negative_zero_has_no_sign():
    assert solution.round_half_up(-0.001, 2) == "0.00"


def test_no_scientific_notation():
    assert solution.round_half_up(1e-7, 3) == "0.000"
    assert solution.round_half_up(1.5e16, 1) == "15000000000000000.0"


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        solution.round_half_up(float("nan"), 2)
    with pytest.raises(ValueError):
        solution.round_half_up(float("inf"), 2)
    with pytest.raises(ValueError):
        solution.round_half_up(1.0, -1)
