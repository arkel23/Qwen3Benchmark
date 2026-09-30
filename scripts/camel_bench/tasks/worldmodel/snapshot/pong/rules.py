"""Stage-1a wall-ball rule family: slots, enumeration, canonical program text and compiled Python."""
import itertools
from dataclasses import astuple, dataclass, fields

SLOTS = {
    "paddle_height": (2, 3, 4),
    "paddle_speed": (1, 2),
    "ball_period": (1, 2),
    "wall": ("bounce", "wrap"),
    "hit_vy": ("keep", "flip", "segment", "paddle"),
    "serve_vy": (-1, 0, 1),
}
HIT_EFFECT = {"keep": "vy = vy", "flip": "vy = -vy", "segment": "vy = segment(-1, 0, +1)",
              "paddle": "vy = paddle.last_move"}


@dataclass(frozen=True)
class Program:
    paddle_height: int
    paddle_speed: int
    ball_period: int
    wall: str
    hit_vy: str
    serve_vy: int

    def index(self):
        return tuple(SLOTS[f.name].index(value) for f, value in zip(fields(self), astuple(self)))

    def text(self):
        wall = "vy = -vy" if self.wall == "bounce" else "y = wrap(y)"
        return "\n".join([
            f"init:             paddle.height = {self.paddle_height} ; ball.vx = -1 ; ball.vy = {self.serve_vy:+d}",
            f"tick:             paddle.move(action * {self.paddle_speed}) ; ball.move every {self.ball_period}",
            f"hit(paddle):      vx = -vx ; {HIT_EFFECT[self.hit_vy]}",
            "hit(wall_right):  vx = -vx",
            f"hit(wall_tb):     {wall}",
            f"exit(left):       misses += 1 ; ball.reset(vx=-1, vy={self.serve_vy:+d})",
        ])

    def python(self):
        """Source of a scalar Python class implementing this program; `step` returns the visible state."""
        hit_vy = {"keep": "self.vy", "flip": "-self.vy",
                  "segment": "sign(2 * (self.y - self.top) - (h - 1))", "paddle": "last_move"}[self.hit_vy]
        wall = ("                self.vy = -self.vy\n                ny = self.y + self.vy" if self.wall == "bounce"
                else "                ny = (ny - 1) % 15 + 1")
        return f'''def sign(v):
    return (v > 0) - (v < 0)


class Game:
    def __init__(self):
        h = {self.paddle_height}
        self.top, self.x, self.y, self.vx, self.vy = 8 - h // 2, 8, 8, -1, {self.serve_vy}
        self.phase, self.misses = 0, 0

    def step(self, action):
        h = {self.paddle_height}
        previous = self.top
        self.top = min(max(self.top + action * {self.paddle_speed}, 1), 16 - h)
        last_move = sign(self.top - previous)
        self.phase = (self.phase + 1) % {self.ball_period}
        if self.phase == 0:
            if self.x == 1 and self.vx == -1 and self.top <= self.y < self.top + h:
                self.vx, self.vy = 1, {hit_vy}
            if self.x == 15 and self.vx == 1:
                self.vx = -1
            nx, ny = self.x + self.vx, self.y + self.vy
            if not 1 <= ny <= 15:
{wall}
            if nx == 0:
                self.misses += 1
                nx, ny, self.vx, self.vy, self.phase = 8, 8, -1, {self.serve_vy}, 0
            self.x, self.y = nx, ny
        return self.top, self.x, self.y, self.misses
'''


def all_programs():
    return [Program(*values) for values in itertools.product(*SLOTS.values())]


def compile_program(program):
    namespace = {}
    exec(program.python(), namespace)
    return namespace["Game"]
