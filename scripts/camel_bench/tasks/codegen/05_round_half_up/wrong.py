import math


def round_half_up(x: float, decimals: int = 2) -> str:
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0, got {decimals}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot round {x}")
    scale = 10 ** decimals
    rounded = math.floor(x * scale + 0.5) / scale
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"
