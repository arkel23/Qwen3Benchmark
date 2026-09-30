I want a PyTorch helper for gradient accumulation so a batch that doesn't fit in memory can be split into micro-batches.

Module-level function:

`accumulation_step(model: nn.Module, optimizer: torch.optim.Optimizer, micro_batches: list[tuple[torch.Tensor, torch.Tensor]], loss_fn) -> float`

- `micro_batches` is a list of `(inputs, targets)` pairs; `loss_fn(model(inputs), targets)` returns a scalar loss tensor (a mean over the micro-batch).
- Call `optimizer.zero_grad()` once at the start, so gradients left from earlier steps do not leak in.
- For each micro-batch, compute the loss, divide it by the number of micro-batches and call `backward()` on the scaled loss.
- After all micro-batches, call `optimizer.step()` exactly once. Do not zero the gradients afterwards: `.grad` must still hold the accumulated gradients when the function returns.
- With equal-sized micro-batches, the accumulated gradients and the updated parameters must match a single step on the concatenated batch.
- Return the mean of the unscaled micro-batch losses as a Python `float`.
- Raise `ValueError` if `micro_batches` is empty, before touching the optimizer.

Reply with a single Python code block containing the complete module.
