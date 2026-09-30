# Code-generation tasks

Twelve tasks ask a model for a small research-code module. Each task folder holds `prompt.md` (the user message), `test_task.py` (hidden tests), `reference.py` (a correct solution) and `wrong.py` (a plausible solution with a subtle bug).

| Task | What the tests check | Tests |
|---|---|---|
| `01_cer` | Levenshtein distance, punctuation and whitespace normalisation, corpus-level CER, errors on mismatched or empty input | 10 |
| `02_topk_accuracy` | Top-k accuracy for several k from torch logits, float outputs, range and shape errors | 8 |
| `03_linear_cka` | Linear CKA with column centring, match to the HSIC form, orthogonal, scale and offset invariance, input errors | 9 |
| `04_paired_bootstrap` | Paired percentile bootstrap CI of the mean difference, exact seeded procedure, determinism, input errors | 8 |
| `05_round_half_up` | Half-away-from-zero rounding to a fixed-decimal string, binary-representation cases, negative zero, errors | 9 |
| `06_latex_table` | DataFrame to booktabs tabular, column spec, LaTeX escaping, min/max bolding with ties and NaN, errors | 9 |
| `07_paired_merge` | Strict one-to-one merge on keys, per-key metric difference, output columns and order, duplicate and mismatch errors | 9 |
| `08_ema_smoothing` | TensorBoard-style EMA with bias correction, NaN skipping and propagation, errors | 9 |
| `09_argparse_cli` | argparse options, types, defaults, choices, `nargs="+"`, flag, `SystemExit` on invalid input | 9 |
| `10_patch_embed` | ViT patch embedding shapes, row-major patch order, CLS token, positional embedding, size errors | 9 |
| `11_grad_accumulation` | Gradient accumulation equals a full-batch step for SGD and Adam, one optimizer step, cleared stale gradients | 7 |
| `12_stratified_split` | Seeded stratified split, per-class floor with a one-sample minimum, disjoint sorted indices, errors | 9 |

The grader writes the model's code block to `solution.py` next to a copy of `test_task.py` and runs `pytest -q`; the score is the fraction of tests passed.
`validate.py` checks every task: `reference.py` passes all tests, while `wrong.py` and an empty module each fail at least one.
