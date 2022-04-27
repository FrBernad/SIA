from typing import List

from numpy import random, vectorize, tanh, exp, copysign, zeros
from numpy.linalg import norm
from numpy.random import rand
from numpy.typing import NDArray

from utils.config import PerceptronConfig, PerceptronSettings

_FUNCTIONS = {
    "tanh": (
        lambda b, h: tanh(b * h),
        lambda b, h: b * (1 - tanh(b * h) ** 2)
    ),
    "logistic": (
        lambda b, h: 1 / (1 + exp(-2 * b * h)),
        lambda b, h: 2 * b * (1 / (1 + exp(-2 * b * h))) * (1 - (1 / (1 + exp(-2 * b * h))))
    )
}


class SimplePerceptron:

    def __init__(
            self,
            x: NDArray[[float, float, float]],
            y: NDArray[float],
            config: PerceptronSettings
    ):
        self.plot = dict(
            e=[],
            o=[]
        )
        self.w = None
        self.x = x
        self.y = y
        self.threshold = config.threshold
        self.learning_rate = config.learning_rate
        self.dimension = len(x[0]) - 1
        self.examples_count = len(x)

    def train(self):

        i = 0
        w = random.uniform(-1, 1, size=self.dimension + 1)
        w_min = w
        error = 1
        error_min = self.examples_count * 2
        o = []

        while error > 0 and i < self.threshold:

            i_x = random.randint(0, self.examples_count)
            h = self.x @ w
            o = vectorize(self.activation_function)(h)

            delta_w = self.delta_w(self.y[i_x], o[i_x], self.x[i_x], h[i_x])
            w += delta_w

            error = self.error(o, self.y)

            self.plot['e'].append(error)

            if error < error_min:
                error_min = error
                w_min = w

            i = i + 1

        self.plot['e'].append(error)
        self.plot['o'] = o
        self.w = w_min

        # print(w_min)

        # print(i)
        print(error_min)

        # print(o)
        # print(self.y)

    def predict(
            self,
            x_d,
            y_d,
            x: NDArray[[float, float, float]],
            y: NDArray[float],
    ):
        x = vectorize(lambda v: (v - x_d.min()) / (x_d.max() - x_d.min()) - 1)(x)
        y = vectorize(lambda v: (v - y_d.min()) / (y_d.max() - y_d.min()) - 1)(y)

        h = x @ self.w
        o = vectorize(self.activation_function)(h)
        error = self.error(o, y)

        print(o)
        print(y)

        print(error)

    def activation_function(
            self,
            h
    ):
        return copysign(1, h)

    def delta_w(
            self,
            y: float,
            o: float,
            x: float,
            h: float
    ):
        return self.learning_rate * (y - o) * x

    def error(
            self,
            o: NDArray[float],
            y: NDArray[float]
    ):
        return 0.5 * sum((y - o) ** 2)


class LinearPerceptron(SimplePerceptron):
    def __init__(
            self,
            x: NDArray[[float, float, float]],
            y: NDArray[float],
            config: PerceptronSettings
    ):
        super().__init__(x, y, config)

    def activation_function(
            self,
            h
    ):
        return h


class NonLinearPerceptron(SimplePerceptron):

    def __init__(
            self,
            x: NDArray[[float, float, float]],
            y: NDArray[float],
            config: PerceptronSettings
    ):
        y = vectorize(lambda v: 2 * (v - y.min()) / (y.max() - y.min()) - 1)(y)  # Normalize Output
        super().__init__(x, y, config)
        self.g = _FUNCTIONS[config.g][0]
        self.g_derivative = _FUNCTIONS[config.g][1]
        self.b = config.b

    def activation_function(
            self,
            h
    ):
        return self.g(self.b, h)

    def delta_w(
            self,
            y: float,
            o: float,
            x: float,
            h: float
    ):
        return self.learning_rate * (y - o) * x * self.g_derivative(self.b, h)


class Perceptron:
    def __init__(self, h: float, v: NDArray[float], d: float, w: NDArray[float]):
        self.h = h
        self.v = v
        self.d = d
        self.w = w


class MultiLayerPerceptron:

    def __init__(
            self,
            x: NDArray,
            neurons_per_layer: List[int],
            y: NDArray,
            config: PerceptronSettings
    ):
        self.x = x
        self.y = y

        self.plot = dict(
            e=[]
        )

        self.g = _FUNCTIONS[config.g][0]
        self.g_derivative = _FUNCTIONS[config.g][1]
        self.b = config.b

        self.neurons_per_layer = neurons_per_layer

        self.w = []
        self.d_j = []
        self.v = []
        self.h_m = []

        for i in range(len(neurons_per_layer) - 1):
            self.w.append(rand(neurons_per_layer[i + 1], neurons_per_layer[i]))
            self.d_j.append(zeros(neurons_per_layer[i + 1]))
            self.v.append(zeros(neurons_per_layer[i]))
            self.h_m.append(zeros(neurons_per_layer[i + 1]))

        self.v.append(zeros(neurons_per_layer[-1]))

        self.threshold = config.threshold
        self.learning_rate = config.learning_rate

    def train(self):

        i = 0
        error = 1
        error_min = len(self.x) * 2
        while error > 0 and i < self.threshold:
            for x_i in range(0, len(self.x)):

                self.v[0] = self.x[x_i]

                self.propagate()
                self.calculate_d_M(self.y[x_i])
                self.retro_propagate()
                self.update_weights()

                error = self.error(self.x, self.y)

                self.plot['e'].append(error)

                if error < error_min:
                    error_min = error

                i = i + 1

        self.plot['e'].append(error)

        print(error_min)

    def predict(self, x):
        for m in range(1, len(self.neurons_per_layer)):
            h = self.w[m - 1] @ x
            x = self.g(self.b, h)
            if m != len(self.neurons_per_layer) - 1:
                x[0] = 1

        return x

    def propagate(self):
        # Iterate layers and calculate Vm
        for m in range(1, len(self.neurons_per_layer)):
            self.h_m[m - 1] = self.w[m - 1] @ self.v[m - 1]
            self.v[m] = self.g(self.b, self.h_m[m - 1])
            # Make sure bias is maintained
            if m != len(self.neurons_per_layer) - 1:
                self.v[m][0] = 1

        # O:       Y
        # V:  x0 x1 x2
        # E: e0  e1  e2

        # w = [[2, 3], [1, 3]]
        # w = [ [ [w00,w01,w02], [w10, w11, w12],[w20,w21,w22] ], [W10,W11,W12] ]
        # V0 = [e0, e1, e2]
        # V[1][0] = [e0*w10+e1*w11+e2*w12]

    def calculate_d_M(self, y):
        self.d_j[-1] = self.g_derivative(self.b, self.h_m[-1]) * (y - self.v[-1])

    def retro_propagate(self):
        for m in range(len(self.h_m) - 1, 0, -1):
            self.d_j[m - 1] = self.g_derivative(self.b, self.w[m].T @ self.d_j[m])

    def update_weights(self):
        for m in range(1, len(self.neurons_per_layer)):
            self.w[m - 1] += self.learning_rate * self.d_j[m - 1] * self.v[m]

    def error(
            self,
            x: NDArray[float],
            y: NDArray[float]
    ):
        o = []
        for value in x:
            o.append(*self.predict(value))

        return 0.5 * sum((y - o) ** 2)
