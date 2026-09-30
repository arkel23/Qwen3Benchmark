import argparse

import pytest

import solution


def test_defaults():
    args = solution.parse(["--model", "vit"])
    assert args.model == "vit"
    assert args.lr == pytest.approx(1e-3)
    assert args.epochs == 10
    assert args.optimizer == "adamw"
    assert args.fp16 is False
    assert args.seed == 42
    assert args.output_dir == "results"


def test_datasets_default_is_list():
    assert solution.parse(["--model", "vit"]).datasets == ["cub"]


def test_types_converted():
    args = solution.parse(["--model", "vit", "--lr", "0.1", "--epochs", "3", "--seed", "7"])
    assert type(args.lr) is float and args.lr == pytest.approx(0.1)
    assert args.epochs == 3 and type(args.epochs) is int
    assert args.seed == 7


def test_datasets_multiple_values():
    args = solution.parse(["--model", "vit", "--datasets", "cub", "dogs", "nabirds"])
    assert args.datasets == ["cub", "dogs", "nabirds"]


def test_flag_and_output_dir():
    args = solution.parse(["--model", "vit", "--fp16", "--output-dir", "out/run1", "--optimizer", "sgd"])
    assert args.fp16 is True
    assert args.output_dir == "out/run1"
    assert args.optimizer == "sgd"


def test_invalid_choice_exits():
    with pytest.raises(SystemExit):
        solution.parse(["--model", "vit", "--optimizer", "lion"])


def test_missing_model_exits():
    with pytest.raises(SystemExit):
        solution.parse([])


def test_non_numeric_lr_exits():
    with pytest.raises(SystemExit):
        solution.parse(["--model", "vit", "--lr", "fast"])


def test_zero_epochs_exits_in_parse_only():
    with pytest.raises(SystemExit):
        solution.parse(["--model", "vit", "--epochs", "0"])
    parser = solution.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.parse_args(["--model", "vit", "--epochs", "0"]).epochs == 0
