from numpy import dot
from numpy.linalg import norm
from numpy.random import rand
from numpy.typing import NDArray

from utils.config import OjaConfig


class Oja:

    def __init__(self, input_values: NDArray, config: OjaConfig):
        self.input_values = input_values
        self.max_iter = config.max_iter
        self.learning_rate = config.learning_rate
        self.n = len(input_values)
        self.w = rand(len(input_values[0]))
        self.w = self.w / norm(self.w)

    def train(self) -> 'OjaResults':
        iteration = 0

        results = OjaResults()
        results.w.append(self.w)

        while iteration < self.max_iter:
            for i in range(self.n):
                s = self.input_values[i] @ self.w

                self.w = self.w / norm(self.w) + self.learning_rate * s * (
                        self.input_values[i] / norm(self.w) - s * self.w / norm(self.w) ** 3)

                results.w.append(self.w)

            iteration += 1

        return results


class OjaResults:
    def __init__(self):
        self.w = []
