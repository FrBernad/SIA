from typing import Callable

from numpy import tanh


def simple_perceptron_delta_w(y: float, o: float, x: float, h: float, learning_rate: float) -> float:
    return learning_rate * (y - o) * x


def simple_linear_perceptron_delta_w(y: float, o: float, x: float, h: float, learning_rate: float) -> float:
    return simple_perceptron_delta_w(y, o, x, learning_rate)


def simple_no_linear_perceptron_delta_w(y: float, o: float, x: float, h: float, learning_rate: float) -> float:
    return learning_rate * (y - o) * x * 0.01 * (1 - tanh(h) ** 2)


DELTA_W_FUNCTIONS = {
    'simple_perceptron': simple_perceptron_delta_w,
    'linear_perceptron': simple_linear_perceptron_delta_w,
    'no_linear_perceptron': simple_no_linear_perceptron_delta_w
}
