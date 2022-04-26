from numpy import random, vectorize, tanh, exp, copysign
from numpy.linalg import norm
from numpy.typing import NDArray

from utils.config import PerceptronConfig, PerceptronSettings


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
        # print(error_min)

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
        self.x = vectorize(lambda v: (v - x.min()) / (x.max() - x.min()) - 1)(x)
        self.y = vectorize(lambda v: (v - y.min()) / (y.max() - y.min()) - 1)(y)

        # self.x = x/norm(x)
        # self.y = y/norm(y)

    def activation_function(
            self,
            h
    ):
        return h


class NonLinearPerceptron(SimplePerceptron):
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

    def __init__(
            self,
            x: NDArray[[float, float, float]],
            y: NDArray[float],
            config: PerceptronSettings
    ):
        super().__init__(x, y, config)
        self.y = vectorize(lambda v: 2 * (v - y.min()) / (y.max() - y.min()) - 1)(y)  # Normalize Output
        self.g = NonLinearPerceptron._FUNCTIONS[config.g][0]
        self.g_derivative = NonLinearPerceptron._FUNCTIONS[config.g][1]
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
