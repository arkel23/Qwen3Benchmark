import math
from decimal import ROUND_HALF_UP, Decimal


def round_half_up(x: float, decimals: int = 2) -> str:
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0, got {decimals}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot round {x}")
    rounded = Decimal(repr(x)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return f"{rounded:f}"
