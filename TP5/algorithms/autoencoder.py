from typing import List

from numpy import concatenate, flip, array, tanh, matmul, sum as npsum, mean
from numpy.random import uniform
from numpy.typing import NDArray
from scipy.optimize import minimize

from utils.config import Config


class Autoencoder:

    def __init__(
            self,
            x: NDArray,
            y: NDArray,
            config: Config
    ):
        self.x = x
        self.y = y
        self.iterations = 0
        self.max_iter = config.max_iter
        self.min_error = config.min_error

        self.activation_function = lambda h: tanh(h)

        self.latent_space = config.latent_layer
        self.intermediate_layers = config.intermediate_layers
        self.network_layers = [len(x[0]), *self.intermediate_layers, config.latent_layer,
                               *flip(self.intermediate_layers), len(x[0])]

        self.dimensions = []

        self.weights = []
        for i in range(len(self.network_layers) - 1):
            self.dimensions.append((self.network_layers[i + 1], self.network_layers[i]))
            self.weights.append(uniform(low=-1, high=1, size=(self.network_layers[i + 1], self.network_layers[i])))

    def train(self):
        self.iterations = 0
        print("\t\tIterations: ", end="")

        minimization_results = minimize(
            self._calculate_error, self._flatten_w(),
            method="Powell", callback=self._iteration_callback,
            options={'xtol': self.min_error, 'maxiter': self.max_iter, 'disp': False}
        )

        print("")

        return AutoencoderResults(self.iterations, self._reshape_weights(minimization_results.x),
                                  minimization_results.fun)

    def _flatten_w(self):
        flattened_w = []
        for w in self.weights:
            flattened_w = concatenate((flattened_w, w.flatten()))
        return flattened_w

    def _calculate_error(
            self,
            w: NDArray[float],
    ):
        o = []
        reshaped_weights = self._reshape_weights(w)
        for value in self.x:
            o.append(self.propagate(reshaped_weights, value))
        o = array(o)
        return mean((npsum((self.y - o) ** 2, axis=1) / 2))

    def _reshape_weights(self, w):
        weights = []
        current_index = 0
        for dim in self.dimensions:
            flattened_dim = dim[0] * dim[1]
            weights.append(w[current_index:current_index + flattened_dim].reshape(dim))
            current_index += flattened_dim

        return weights

    def _iteration_callback(self, xk):
        self.iterations += 1
        print(f"{self.iterations}", end=" ")

    def propagate(self, weights: List, input_value: NDArray) -> NDArray:
        activation_value = input_value.reshape((len(input_value), 1))
        for layer in range(len(weights)):
            activation_value = self.activation_function(matmul(weights[layer], activation_value))
        return activation_value.flatten()

    def encode(self, input_value, weights):
        return self.propagate(weights[:int(len(weights) / 2)], input_value)

    def decode(self, input_latent_value, weights):
        return self.propagate(weights[int(len(weights) / 2):], input_latent_value)


class AutoencoderResults:
    def __init__(self, iterations, weights, error):
        self.weights = weights
        self.iterations = iterations
        self.error = error
