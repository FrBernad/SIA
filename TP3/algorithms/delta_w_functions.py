from typing import Callable


def simple_perceptron_delta_w(y: float, o: float, x: float, learning_rate: float) -> float:
    return learning_rate * (y - o) * x


def simple_linear_perceptron_delta_w(y: float, o: float, x: float, learning_rate: float) -> float:
    return simple_perceptron_delta_w(y, o, x, learning_rate)


def simple_no_linear_perceptron_delta_w(y: float, o: float, x: float, learning_rate: float, g: Callable) -> float:
    return learning_rate * (y - o) * x


DELTA_W_FUNCTIONS = {
    'simple_perceptron': simple_perceptron_delta_w,
    'linear_perceptron': simple_linear_perceptron_delta_w,
    'no_liner_perceptron': simple_no_linear_perceptron_delta_w
}
