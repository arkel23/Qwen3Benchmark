For paper tables I need numbers rounded the way people expect (half away from zero), not Python's banker's rounding and not affected by binary float representation. Standard library only.

Module-level function:

`round_half_up(x: float, decimals: int = 2) -> str`

- Interpret `x` by its shortest decimal representation, `repr(float(x))`, so `2.675` is treated as exactly 2.675 and `1.005` as exactly 1.005.
- Round to `decimals` places with ties going away from zero: `0.125 -> "0.13"`, `2.675 -> "2.68"`, `-2.5` with 0 decimals `-> "-3"`.
- Return a string with exactly `decimals` digits after the decimal point, padding with zeros (`1.5` with 3 decimals `-> "1.500"`). With `decimals=0` there is no decimal point (`2.5 -> "3"`). Never use scientific notation.
- A result that rounds to zero has no minus sign: `-0.001` with 2 decimals `-> "0.00"`.
- Raise `ValueError` if `x` is NaN or infinite, or if `decimals` is negative.

Reply with a single Python code block containing the complete module.
