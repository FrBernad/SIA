from typing import List

from numpy import concatenate, flip, array, tanh, matmul, sum as npsum, mean
from numpy.random import uniform
from numpy.typing import NDArray
from scipy.optimize import minimize

from utils.config import AEConfig


class Autoencoder:

    def __init__(
            self,
            x: NDArray,
            intermediate_layers: List[int],
            latent_space: int,
            config: AEConfig
    ):
        self.iterations = 0
        self.x = x

        self.max_iter = config.max_iter
        self.min_error = config.min_error

        self.activation_function = lambda h: tanh(h)

        self.latent_space = latent_space
        self.intermediate_layers = intermediate_layers
        self.network_layers = [len(x[0]), *intermediate_layers, latent_space, *flip(intermediate_layers), len(x[0])]

        self.dimensions = []

        self.w = []
        for i in range(len(self.network_layers) - 1):
            self.dimensions.append((self.network_layers[i + 1], self.network_layers[i]))
            self.w.append(uniform(low=-1, high=1, size=(self.network_layers[i + 1], self.network_layers[i])))

    def train(self):
        results = minimize(
            self.calculate_error, self._flatten_w(),
            method="Powell", callback=self.iteration_callback,
            options={'xtol': self.min_error, 'maxiter': self.max_iter}
        )
        print(self.propagate(self.reshape_weights(results.x), self.x[0]))
        print(self.x[0])

    def iteration_callback(self, xk):
        self.iterations += 1
        print(f"iteration {self.iterations}")

    def _flatten_w(self):
        flattened_w = []
        for w in self.w:
            flattened_w = concatenate((flattened_w, w.flatten()))
        return flattened_w

    def calculate_error(
            self,
            w: NDArray[float],
    ):
        o = []
        reshaped_weights = self.reshape_weights(w)
        for value in self.x:
            o.append(self.propagate(reshaped_weights, value))
        o = array(o)
        return mean((npsum((self.x - o) ** 2, axis=1) / 2))

    def reshape_weights(self, w):
        weights = []
        current_index = 0
        for dim in self.dimensions:
            flattened_dim = dim[0] * dim[1]
            weights.append(w[current_index:current_index + flattened_dim].reshape(dim))
            current_index += flattened_dim

        return weights

    def propagate(self, weights: List, output: NDArray) -> NDArray:
        activation_value = output.reshape((len(output), 1))
        for layer in range(len(weights)):
            activation_value = self.activation_function(matmul(weights[layer], activation_value))
        return activation_value.flatten()
