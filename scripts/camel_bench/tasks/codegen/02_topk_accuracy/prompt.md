Can you write a PyTorch helper for top-k classification accuracy?

Module-level function:

`topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> dict[int, float]`

- `logits` has shape `(N, C)`; `targets` has shape `(N,)` with integer class indices in `[0, C)`.
- A sample counts as correct at k if its target is among the k classes with the largest logits.
- Return a dict mapping each k in `ks` to the fraction of correct samples (a Python `float` in `[0, 1]`, not a tensor). The dict keys are exactly the values in `ks`.
- It must work when `logits.requires_grad` is True and must not build an autograd graph.
- Raise `ValueError` if `logits` is not 2-D, `targets` is not 1-D, `N` differs between them, or `N == 0`.
- Raise `ValueError` if any k is smaller than 1 or larger than `C`.

Reply with a single Python code block containing the complete module.
