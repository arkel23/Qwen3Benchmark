#!/usr/bin/env bash
# One-off commands, dated. Dry run by default; pass --execute to run.
set -euo pipefail
cd "$(dirname "$0")"
EXECUTE=${1:-}

run() {
    echo "+ $*"
    if [[ "$EXECUTE" == "--execute" ]]; then "$@"; fi
}

# 2026-09-18: rule coverage of one controller rollout per stage-1a program (FINDINGS.md, rule coverage).
run python -c "
import numpy as np
from pong.rules import all_programs
from pong.sim import simulate
programs = all_programs()
for ticks in (128, 256):
    visible, _ = simulate(programs, ticks, seeds=np.arange(len(programs)))
    x, misses = visible[..., 1], visible[..., 3]
    hits = ((x[:, :-1] == 1) & (x[:, 1:] == 2)).sum(1)
    print(ticks, (hits > 0).mean(), (misses[:, -1] > 0).mean(), np.median(hits), np.median(misses[:, -1]))
"
