def accumulation_step(model, optimizer, micro_batches, loss_fn) -> float:
    if not micro_batches:
        raise ValueError("micro_batches is empty")
    optimizer.zero_grad()
    total = 0.0
    for inputs, targets in micro_batches:
        loss = loss_fn(model(inputs), targets)
        (loss / len(micro_batches)).backward()
        total += loss.item()
    optimizer.step()
    return total / len(micro_batches)
