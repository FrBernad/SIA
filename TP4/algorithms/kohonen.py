from math import exp

from numpy import zeros, unravel_index, argmin, array, copy, average
from numpy.linalg import norm
from numpy.random import randint
from numpy.typing import NDArray

from utils.config import KohonenConfig


class Kohonen:

    def __init__(self, input_values: NDArray, config: KohonenConfig):
        self.input_values = input_values
        self.radius = config.radius
        self.init_radius = config.radius
        self.k = config.k
        self.max_iter = config.max_iter
        self.learning_rate = config.learning_rate
        self.init_learning_rate = config.learning_rate

        self.neurons = zeros((self.k, self.k, len(input_values[0])))

        for i in range(self.k):
            for j in range(self.k):
                self.neurons[i][j] = copy(input_values[randint(len(input_values))])

    def train(self):

        iteration = 1
        while iteration <= self.max_iter:
            x = self.input_values[randint(len(self.input_values))]

            winner = self.get_winner(x)
            self._update_neighbors(x, winner)

            iteration += 1

            self.learning_rate = self.init_learning_rate * exp(-iteration / self.max_iter)

            self.radius = self.init_radius * exp(-iteration / self.max_iter)
            if self.radius < 1:
                self.radius = 1

    def _update_neighbors(self, x, winner):
        for i in range(self.k):
            for j in range(self.k):
                if norm(array([i, j]) - winner) <= self.radius:
                    self.neurons[i][j] += self.learning_rate * (x - self.neurons[i][j])

    def get_winner(self, x: NDArray):
        norms = norm(self.neurons - x, axis=2)
        return array(unravel_index(argmin(norms, axis=None), norms.shape))

    def get_neighbors_avg(self, neuron: NDArray):
        euc_distance = []
        radius = 1.5
        for i in range(self.k):
            for j in range(self.k):
                if norm(array([i, j]) - neuron) <= radius:
                    euc_distance.append(norm(self.neurons[i][j] - self.neurons[neuron[0]][neuron[1]]))

        return average(euc_distance)
