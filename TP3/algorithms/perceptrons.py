from typing import List, Optional

from numpy import random, vectorize, tanh, exp, copysign, array
from numpy.random import randint
from numpy.typing import NDArray

from utils.config import PerceptronSettings

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
        self.min_iter = config.min_iter
        self.min_error = config.min_error
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

        while error > self.min_error and i < self.min_iter:

            i_x = random.randint(0, self.examples_count)
            h = self.x @ w
            o = []
            for val in h:
                o.append([self.activation_function(val)])
            o = array(o)

            delta_w = self.delta_w(self.y[i_x], o[i_x], self.x[i_x], h[i_x])
            w += delta_w

            error = self.error(o, self.y)

            self.plot['e'].append(error[0])

            if error < error_min:
                error_min = error
                w_min = w

            i = i + 1

        self.plot['e'].append(error)
        self.plot['o'] = o
        self.w = w_min

        print(error_min)

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

        self.neurons_per_layer = [len(x[0]), *hidden_layers, len(y[0])]
        self.layers_count = len(self.neurons_per_layer)
        self.layers = []

        for i in range(self.layers_count):
            layer = []
            for n in range(self.neurons_per_layer[i]):
                perceptron = Perceptron(None, None, None, None)
                if i != 0 and (n != self.neurons_per_layer[i] - 1 or i == self.layers_count - 1):
                    perceptron.w = random.uniform(-1, 1, size=self.neurons_per_layer[i - 1])
                    perceptron.v = 0
                    perceptron.d = 0
                    perceptron.h = 0
                layer.append(perceptron)
            self.layers.append(layer)

        self.min_iter = config.min_iter
        self.min_error = config.min_error
        self.learning_rate = config.learning_rate

    def train(self):

        i = 0
        error = 1
        error_min = len(self.x) * 2
        while error > self.min_error and i < self.min_iter:
            i_x = randint(0, len(self.x))

            self.apply_first_layer(i_x)
            self.propagate()
            self.calculate_d_M(self.y[i_x])
            self.retro_propagate()

            error = self.calculate_error(self.x, self.y)[0]
            self.plot['e'].append(error)
            if error < error_min:
                error_min = error

            self.update_weights()

            i = i + 1

        o = []
        for value in self.x:
            o.append(self.predict(value))
        self.plot['e'].append(error)
        print(i)
        print(error_min)

    def apply_first_layer(self, i_x):
        for i in range(self.neurons_per_layer[0]):
            self.layers[0][i].v = self.x[i_x][i]

    def propagate(self):
        # Para cada capa oculta hasta la de salida
        for m in range(1, self.layers_count):
            # Para cada neurona i del nivel m
            for i in range(self.neurons_per_layer[m]):
                self.layers[m][i].h = 0
                # Si no es el umbral o la ultima capa
                if i != self.neurons_per_layer[m] - 1 or m == self.layers_count - 1:
                    # Para cada neurona de la capa de abajo sumo el peso
                    for j in range(self.neurons_per_layer[m - 1]):
                        self.layers[m][i].h += self.layers[m][i].w[j] * self.layers[m - 1][j].v
                    self.layers[m][i].v = self.g(self.b, self.layers[m][i].h)
                # Si es el umbral
                else:
                    self.layers[m][i].v = 1

    def calculate_d_M(self, y):
        for i in range(self.neurons_per_layer[-1]):
            perceptron = self.layers[-1][i]
            perceptron.d = self.g_derivative(self.b, perceptron.h) * (y - perceptron.v)
            # O:       Y
            # Vj:  x1 x2 x0
            # V0: e1  e2  e0

    def retro_propagate(self):
        # Por cada capa oculta de arriba para abajo
        for m in range(self.layers_count - 1, 1, -1):
            # Por cada neurona de la capa m-1
            for i in range(self.neurons_per_layer[m - 1] - 1):
                aux = 0
                # Por cada peso que sale de la neurona m-1
                for j in range(self.neurons_per_layer[m]):
                    if j != self.neurons_per_layer[m] - 1 or m == self.layers_count - 1:
                        aux += self.layers[m][j].w[i] * self.layers[m][j].d

                self.layers[m - 1][i].d = self.g_derivative(self.b, self.layers[m - 1][i].h) * aux

    def update_weights(self):
        # para cada capa de abajo hacia arriba
        for m in range(1, self.layers_count):
            # para cada neurona i del nivel m
            for i in range(self.neurons_per_layer[m]):
                # Para cada neurona de la capa m-1
                if i != self.neurons_per_layer[m] - 1 or m == self.layers_count - 1:
                    for j in range(self.neurons_per_layer[m - 1]):
                        self.layers[m][i].w[j] += self.learning_rate * self.layers[m][i].d * self.layers[m - 1][j].v

    def calculate_error(
            self,
            x: NDArray[float],
            y: NDArray[float]
    ):
        o = []
        for value in x:
            o.append([self.predict(value)])
        o = array(o)
        return 0.5 * sum((y - o) ** 2)

    def predict(self, x):
        layers = []

        for i in range(self.layers_count):
            layer = []
            for n in range(self.neurons_per_layer[i]):
                perceptron = Perceptron(None, None, None, None)
                if i != 0 and (n != self.neurons_per_layer[i] - 1 or i == self.layers_count - 1):
                    perceptron.v = 0
                    perceptron.d = 0
                    perceptron.h = 0
                layer.append(perceptron)
            layers.append(layer)

        for i in range(self.neurons_per_layer[0]):
            layers[0][i].v = x[i]

        for m in range(1, self.layers_count):
            for i in range(self.neurons_per_layer[m]):
                if i != self.neurons_per_layer[m] - 1 or m == self.layers_count - 1:
                    for j in range(self.neurons_per_layer[m - 1]):
                        layers[m][i].h += self.layers[m][i].w[j] * layers[m - 1][j].v
                    layers[m][i].v = self.g(self.b, layers[m][i].h)
                else:
                    layers[m][i].v = 1

        return layers[-1][0].v
