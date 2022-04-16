from numpy import zeros, sign, dot
from numpy import random
from numpy.typing import NDArray


def simple_perceptron(
        dimension: int,
        p: int,
        x: NDArray[[float, float]],
        y: NDArray[float],
        u: float,
        threshold: int
):
    i = 0
    w = zeros((dimension + 1, 1))
    error = 1
    error_min = p * 2

    while error > 0 and i < threshold:
        i_x = random.randint(1, p)
        h = dot(x[i_x], w)
        o = sign(h)
        delta_w = dot(u, (dot(y[i_x] - o), x[i_x]))
        w = w + delta_w
        error = calculate_error(x, y, w, p)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1


def calculate_error(
        x: NDArray[[float, float]],
        y: NDArray[float],
        w: NDArray[float],
        p: int
):
    return 1
