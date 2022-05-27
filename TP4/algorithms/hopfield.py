from copy import copy

from numpy import fill_diagonal, matmul, sign, array_equal, argwhere, triu
from numpy.typing import NDArray

from utils.config import HopfieldConfig


class Hopfield:

    def __init__(self, config: HopfieldConfig, patterns: NDArray):
        self.patterns = copy(patterns)
        self.w = (1 / len(patterns)) * matmul(patterns.T, patterns)
        self.max_iter = config.max_iter
        fill_diagonal(self.w, 0)

    def train(self, pattern: NDArray) -> 'HopfieldResults':
        stable = False
        results = HopfieldResults()

        pattern = copy(pattern).reshape((len(pattern), 1))

        s1 = sign(matmul(self.w, pattern))
        _replace_zeros(s1, pattern)

        results.H.append(self._calculate_energy(pattern))
        results.H.append(self._calculate_energy(s1))

        results.patterns.append(s1.reshape(len(s1)))

        iteration = 0
        while not stable and iteration < self.max_iter:
            s2 = sign(matmul(self.w, s1))
            _replace_zeros(s2, s1)

            results.H.append(self._calculate_energy(s2))

            results.patterns.append(s2.reshape(len(s2)))

            if array_equal(s1, s2):
                stable = True

            s1 = s2
            iteration += 1

        return results

    def _calculate_energy(self, s1):
        return -matmul(s1.T, matmul(triu(self.w), s1))[0][0]


def _replace_zeros(s1, pattern):
    for indexes in argwhere(s1 == 0):
        s1[indexes[0], indexes[1]] = pattern[indexes[0], indexes[1]]


class HopfieldResults:
    def __init__(self):
        self.patterns = []
        self.H = []
