import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> dict:
    if logits.dim() != 2 or targets.dim() != 1 or logits.shape[0] != targets.shape[0] or logits.shape[0] == 0:
        raise ValueError(f"bad shapes {tuple(logits.shape)} and {tuple(targets.shape)}")
    if any(k < 1 or k > logits.shape[1] for k in ks):
        raise ValueError(f"ks must lie in [1, {logits.shape[1]}]")
    with torch.no_grad():
        pred = logits.topk(max(ks), dim=1).indices
        correct = pred.eq(targets.unsqueeze(1))
        return {k: correct[:, :k].any(dim=1).float().mean().item() for k in ks}
