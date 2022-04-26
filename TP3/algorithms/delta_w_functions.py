from utils.config import NeuronConfig


def simple_perceptron_delta_w(y: float, o: float, x: float, h: float, config: NeuronConfig) -> float:
    return config.learning_rate * (y - o) * x


def simple_linear_perceptron_delta_w(y: float, o: float, x: float, h: float, config: NeuronConfig) -> float:
    return simple_perceptron_delta_w(y, o, x, h, config)


def simple_non_linear_perceptron_delta_w(y: float, o: float, x: float, h: float, config: NeuronConfig) -> float:
    return config.learning_rate * (y - o) * x * config.g[1](config.b, h)


DELTA_W_FUNCTIONS = {
    'simple_perceptron': simple_perceptron_delta_w,
    'linear_perceptron': simple_linear_perceptron_delta_w,
    'non_linear_perceptron': simple_non_linear_perceptron_delta_w
}
