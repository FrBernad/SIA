from numpy import fill_diagonal, matmul, sign, array_equal
from numpy.typing import NDArray


class Hopfield:

    def __init__(self, patterns: NDArray):
        self.patterns = patterns
        self.w = (1 / len(patterns)) * matmul(patterns.T, patterns)
        fill_diagonal(self.w, 0)

    def train(self, pattern: NDArray) -> 'HopfieldResults':
        stable = False
        results = HopfieldResults()

        pattern = pattern.reshape((len(pattern), 1))

        s1 = sign(matmul(self.w, pattern))

        results.patterns.append(s1)

        while not stable:
            s2 = sign(matmul(self.w, s1))
            results.patterns.append(s2)
            if array_equal(s1, s2):
                stable = True

            s1 = s2

        return results


class HopfieldResults:
    def __init__(self):
        self.patterns = []
