"""Train and evaluate E1 (from-scratch video -> program) on a class split; writes metrics.json to --out.
Run from the repo root: python scripts/train_e1.py --split combination --out <dir>"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pong.classes import consistent_programs
from pong.data import SLOT_SIZES, SPLITS, Family
from pong.e1 import ProgramFromVideo
from pong.torch_render import Renderer

INPUTS = ("full", "actions_only", "first_frame")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--input", choices=INPUTS, default="full")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--ticks", type=int, default=256)
    p.add_argument("--cell", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_seeds", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def to_frames(renderer, visible, heights, actions, mode, device):
    frames = renderer(torch.from_numpy(visible).to(device), torch.from_numpy(heights).to(device))
    if mode == "actions_only":
        frames = torch.zeros_like(frames)
    elif mode == "first_frame":
        frames[:, :, 1:] = 0
    return frames, torch.from_numpy(actions).to(device)


def evaluate(model, family, classes, args, renderer, device, rng):
    """Behavioural equivalence (top-1 and with execution filtering), per-slot accuracy and the prior baseline."""
    model.eval()
    tasks = np.repeat(classes, args.eval_seeds)
    top1, filtered, prior, slots = [], {8: [], 64: []}, {8: [], 64: []}, []
    for start in range(0, len(tasks), args.batch):
        batch = tasks[start:start + args.batch]
        visible, actions, heights = family.batch(batch, args.rollouts, args.ticks, rng)
        frames, acts = to_frames(renderer, visible, heights, actions, args.input, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = [l.float() for l in model(frames, acts)]
        log_probs = [F.log_softmax(l, dim=-1).cpu().numpy() for l in logits]
        predicted = np.stack([lp.argmax(-1) for lp in log_probs], axis=1)
        slots.append(predicted == family.targets(batch))
        top1.extend(family.class_of_slots(predicted) == family.class_of[family.canonical[batch]])
        for i, cls in enumerate(batch):
            truth = family.class_of[family.canonical[cls]]
            for k in filtered:
                draws = np.stack([rng.choice(n, size=k, p=np.exp(lp[i])) for n, lp in zip(SLOT_SIZES, log_probs)], 1)
                draws = np.unique(np.vstack([predicted[i], draws]), axis=0)
                ok = consistent_programs([family.programs[family.index_of[tuple(d)]] for d in draws],
                                         actions[i], visible[i])
                score = np.array([sum(lp[i, s] for lp, s in zip(log_probs, d)) for d in draws])
                best = draws[np.argmax(np.where(ok, score, -np.inf))] if ok.any() else predicted[i]
                filtered[k].append(family.class_of_slots(best[None])[0] == truth)
                sample = rng.choice(len(family.programs), size=k, replace=False)
                ok = consistent_programs([family.programs[j] for j in sample], actions[i], visible[i])
                prior[k].append(ok.any() and family.class_of[sample[np.argmax(ok)]] == truth)
    model.train()
    return {
        "tasks": len(tasks),
        "equivalence_top1": float(np.mean(top1)),
        **{f"equivalence_filtered_k{k}": float(np.mean(v)) for k, v in filtered.items()},
        **{f"prior_filtered_k{k}": float(np.mean(v)) for k, v in prior.items()},
        "slot_accuracy": np.concatenate(slots).mean(axis=0).round(4).tolist(),
    }


def majority_baseline(family, train, test):
    majority = [np.bincount(col).argmax() for col in family.targets(train).T]
    cls = family.class_of_slots(np.array([majority]))[0]
    return float(np.mean(family.class_of[family.canonical[test]] == cls))


def main():
    args = parse()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    family = Family()
    train, test = family.split(args.split, seed=args.seed)
    renderer = Renderer(args.cell, device)
    model = ProgramFromVideo().to(device)
    visible, actions, heights = family.batch(train[:1], args.rollouts, args.ticks, rng)
    model(*to_frames(renderer, visible, heights, actions, args.input, device))  # sizes the lazy layer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler()
    print(f"{args.split}/{args.input}: {len(train)} train classes, {len(test)} test classes, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    start = time.time()
    for step in range(1, args.steps + 1):
        classes = rng.choice(train, size=args.batch)
        visible, actions, heights = family.batch(classes, args.rollouts, args.ticks, rng)
        frames, acts = to_frames(renderer, visible, heights, actions, args.input, device)
        targets = torch.from_numpy(family.targets(classes)).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(frames, acts)
        loss = sum(F.cross_entropy(l.float(), targets[:, i]) for i, l in enumerate(logits))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.4f} {time.time() - start:.0f}s "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    eval_rng = np.random.default_rng(10_000 + args.seed)
    metrics = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "test": evaluate(model, family, test, args, renderer, device, eval_rng),
        "train_classes_new_seeds": evaluate(model, family, train[:len(test)], args, renderer, device, eval_rng),
        "majority_baseline_test": majority_baseline(family, train, test),
        "seconds": round(time.time() - start),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
