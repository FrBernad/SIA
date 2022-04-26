from math import copysign

from utils.config import NeuronConfig


def simple_perceptron_activation(h: float, config: NeuronConfig) -> float:
    return copysign(1, h)


def simple_linear_perceptron_activation(h: float, config: NeuronConfig) -> float:
    return h


def simple_non_linear_perceptron_activation(h: float, config: NeuronConfig) -> float:
    return config.g[0](config.b, h)


ACTIVATION_FUNCTIONS = {
    'simple_perceptron': simple_perceptron_activation,
    'linear_perceptron': simple_linear_perceptron_activation,
    'non_linear_perceptron': simple_non_linear_perceptron_activation
}
