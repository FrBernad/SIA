from typing import List, Optional

from numpy import random, vectorize, tanh, exp, copysign, zeros
from numpy.linalg import norm
from numpy.random import rand, randint
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
    def __init__(self, h, v, d: float, w: Optional[NDArray[float]]):
        self.h = h
        self.v = v
        self.d = d
        self.w = w


class MultiLayerPerceptron:

    def __init__(
            self,
            x: NDArray,
            hidden_layers: List[int],
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

        self.neurons_per_layer = [len(x[0]), *hidden_layers, 1]
        self.layers_count = len(self.neurons_per_layer)
        self.layers = []

        for i in range(self.layers_count):
            layer = []
            for n in range(self.neurons_per_layer[i]):
                perceptron = Perceptron(0, 0, 0, None)
                if i != self.layers_count - 1:
                    perceptron.w = rand(self.neurons_per_layer[i + 1] - 1)
                layer.append(perceptron)
            self.layers.append(layer)

        self.threshold = config.threshold
        self.learning_rate = config.learning_rate

    def train(self):

        i = 0
        error = 1
        error_min = len(self.x) * 2
        while error > 0 and i < self.threshold:
            i_x = randint(0, len(self.x))

            self.apply_first_layer(i_x)
            self.propagate()
            self.calculate_d_M(self.y[i_x])
            self.retro_propagate()
            self.update_weights()

            error = self.error(self.x, self.y)

            self.plot['e'].append(error)

            if error < error_min:
                error_min = error

            i = i + 1

        self.plot['e'].append(error)

        print(error_min)

    def apply_first_layer(self, i_x):
        for i in range(self.neurons_per_layer[0]):
            self.layers[0][i].v = self.x[i_x][i]

    def propagate(self):
        # Para cada capa oculta hasta la de salida
        for m in range(1, self.layers_count):
            # Para cada neurona i del nivel m
            for i in range(self.neurons_per_layer[m]):
                if i != 0 or m == self.layers_count - 1:
                    # Para cada neurona de la capa de abajo le pido el peso que le llega a la neurona i
                    for j in range(self.neurons_per_layer[m - 1]):
                        self.layers[m][i].h += self.layers[m - 1][j].w[i] * self.layers[m - 1][j].v
                    self.layers[m][i].v = self.g(self.b, self.layers[m][i].h)
                else:
                    self.layers[m][i].v = 1

    def calculate_d_M(self, y):
        for i in range(self.neurons_per_layer[-1]):
            perceptron = self.layers[-1][i]
            perceptron.d = self.g_derivative(self.b, perceptron.h) * (y - perceptron.v)

    def retro_propagate(self):

        # Por cada capa oculta de arriba para abajo
        for m in range(self.layers_count - 1, 1, -1):
            # Por cada neurona de la capa m-1
            for i in range(self.neurons_per_layer[m - 1]):
                aux = 0
                # O:       Y
                # Vj:  x0 x1 x2
                # V0: e0  e1  e2
                # Por cada conexion de la capa que le llega a la neurona
                for j in range(len(self.layers[m - 1][i].w)):
                    aux += self.layers[m - 1][i].w[j] * self.layers[m][j].d

                self.layers[m - 1][i].d = self.g_derivative(self.b, self.layers[m - 1][i].h) * aux

    def update_weights(self):
        # para cada capa de abajo hacia arriba
        for m in range(0, self.layers_count - 1):
            # para cada neurona i del nivel m
            for i in range(self.neurons_per_layer[m]):
                # Para cada peso que sale de la neurona i
                for j in range(len(self.layers[m][i].w)):
                    self.layers[m][i].w[j] += self.learning_rate * self.layers[m + 1][j].d * self.layers[m][i].v

    def error(
            self,
            x: NDArray[float],
            y: NDArray[float]
    ):
        o = []
        for value in x:
            o.append(self.predict(value))
        return 0.5 * sum((y - o) ** 2)

    def predict(self, x):
        layers = []
        for i in range(self.layers_count):
            layer = []
            for n in range(self.neurons_per_layer[i]):
                perceptron = Perceptron(0, 0, 0, None)
                if i != self.layers_count - 1:
                    perceptron.w = rand(self.neurons_per_layer[i + 1])
                layer.append(perceptron)
            layers.append(layer)

        for i in range(self.neurons_per_layer[0]):
            layers[0][i].v = x[i]

        for m in range(1, self.layers_count):
            for i in range(self.neurons_per_layer[m]):
                if i != 0:
                    for j in range(self.neurons_per_layer[m - 1]):
                        layers[m][i].h += layers[m - 1][j].w[i] * layers[m - 1][j].v
                    layers[m][i].v = self.g(self.b, layers[m][i].h)
                else:
                    layers[m][i].v = 1

        return layers[-1][0].v
