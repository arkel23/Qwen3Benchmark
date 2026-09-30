import numpy as np
import pytest

import solution


def _labels():
    return np.array(["a"] * 5 + ["b"] * 10 + ["c"] * 2 + ["d"] * 1 + ["e"] * 7)


def test_partition_of_all_indices():
    labels = _labels()
    train, val = solution.stratified_split(labels, 0.3, seed=0)
    assert np.intersect1d(train, val).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(labels.size))


def test_per_class_validation_counts():
    labels = _labels()
    _, val = solution.stratified_split(labels, 0.3, seed=0)
    counts = {c: int((labels[val] == c).sum()) for c in "abcde"}
    assert counts == {"a": 1, "b": 3, "c": 1, "d": 0, "e": 2}


def test_singleton_class_in_train():
    labels = _labels()
    train, _ = solution.stratified_split(labels, 0.5, seed=1)
    assert 17 in train


def test_deterministic_given_seed():
    labels = np.repeat(np.arange(4), 25)
    first = solution.stratified_split(labels, 0.2, seed=5)
    second = solution.stratified_split(labels, 0.2, seed=5)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_different_seeds_differ():
    labels = np.repeat(np.arange(4), 25)
    _, val0 = solution.stratified_split(labels, 0.2, seed=0)
    _, val1 = solution.stratified_split(labels, 0.2, seed=1)
    assert not np.array_equal(val0, val1)


def test_sorted_int64_outputs():
    train, val = solution.stratified_split([2, 0, 1, 0, 2, 1, 1, 0, 2, 2], 0.4, seed=3)
    for arr in (train, val):
        assert arr.dtype == np.int64
        np.testing.assert_array_equal(arr, np.sort(arr))


def test_int_labels_proportions():
    labels = np.repeat([0, 1], [20, 40])
    _, val = solution.stratified_split(labels, 0.25, seed=0)
    assert (labels[val] == 0).sum() == 5
    assert (labels[val] == 1).sum() == 10


def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        solution.stratified_split([0, 0, 1, 1], 0.0)
    with pytest.raises(ValueError):
        solution.stratified_split([0, 0, 1, 1], 1.0)


def test_invalid_labels_raise():
    with pytest.raises(ValueError):
        solution.stratified_split(np.zeros((4, 2)), 0.5)
    with pytest.raises(ValueError):
        solution.stratified_split([], 0.5)
