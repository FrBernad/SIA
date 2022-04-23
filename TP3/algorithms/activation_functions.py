from math import copysign, tanh


def simple_perceptron_activation(h: float) -> float:
    return copysign(1, h)


def simple_linear_perceptron_activation(h: float) -> float:
    return h


def simple_no_linear_perceptron_activation(h: float) -> float:
    return tanh(0.01 * h)


ACTIVATION_FUNCTIONS = {
    'simple_perceptron': simple_perceptron_activation,
    'linear_perceptron': simple_linear_perceptron_activation,
    'no_linear_perceptron': simple_no_linear_perceptron_activation
}
