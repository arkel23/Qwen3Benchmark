Please write the command-line interface for a fine-tuning script using `argparse` (standard library only).

Module-level functions:

1. `build_parser() -> argparse.ArgumentParser` returning a parser with these options:

   | Option | Type | Default | Notes |
   |---|---|---|---|
   | `--model` | `str` | none | required |
   | `--lr` | `float` | `1e-3` | |
   | `--epochs` | `int` | `10` | |
   | `--optimizer` | `str` | `"adamw"` | choices `sgd`, `adam`, `adamw` |
   | `--datasets` | `str`, one or more values (`nargs="+"`) | `["cub"]` | always a list |
   | `--fp16` | flag (`store_true`) | `False` | |
   | `--seed` | `int` | `42` | |
   | `--output-dir` | `str` | `"results"` | attribute name `output_dir` |

   The parser itself does no extra validation.

2. `parse(argv: list[str] | None = None) -> argparse.Namespace`
   Builds the parser, parses `argv` (or `sys.argv[1:]` when `None`), and then validates that `epochs >= 1`; if not, it calls `parser.error(...)`, which exits with `SystemExit`. Invalid choices, a missing `--model` and non-numeric values for numeric options also end in `SystemExit` through argparse.

Reply with a single Python code block containing the complete module.
