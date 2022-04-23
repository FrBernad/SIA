from math import copysign
from typing import Callable

from numpy.typing import NDArray


def simple_perceptron_activation(h: float) -> float:
    return copysign(1, h)

def simple_linear_perceptron_activation(weights: NDArray[float], zetas: NDArray[float]) -> float:
    activation = 0
    for i in range(0, zetas.size):
        activation += (weights[i] * zetas[i])
    return activation


def simple_no_linear_perceptron_activation(weights: NDArray[float], zetas: NDArray[float], g: Callable) -> float:
    return g(simple_linear_perceptron_activation(weights, zetas))


ACTIVATION_FUNCTIONS = {
    'simple_perceptron': simple_perceptron_activation,
    'linear_perceptron': simple_linear_perceptron_activation,
    'no_liner_perceptron': simple_no_linear_perceptron_activation
}
