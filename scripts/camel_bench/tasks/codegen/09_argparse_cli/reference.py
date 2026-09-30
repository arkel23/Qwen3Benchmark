import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a backbone.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    parser.add_argument("--datasets", nargs="+", default=["cub"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser


def parse(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    return args
