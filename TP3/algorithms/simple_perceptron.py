import math
from typing import Callable

from numpy import random, vectorize, tanh
from numpy.typing import NDArray

# FIXME: ANIMAR W
from utils.config import NeuronConfig


def simple_perceptron_algorithm(
        x: NDArray[[float, float, float]],
        y: NDArray[float],
        activation_function: Callable,
        delta_w_function: Callable,
        dimension: int,
        examples_count: int,
        config: NeuronConfig
):
    i = 0
    w = random.uniform(size=dimension + 1)
    w_min = w
    error = 1
    error_min = examples_count * 2

    if config.normalize:
        # y = vectorize(lambda v: 2 * (v - y.min()) / (y.max() - y.min()) - 1)(y)
        y = tanh(y)

    while error > 0 and i < config.threshold:
        i_x = random.randint(0, examples_count)
        h = x @ w
        o = vectorize(activation_function)(h, config)

        delta_w = delta_w_function(y[i_x], o[i_x], x[i_x], h[i_x], config)
        w += delta_w
        error = calculate_error(y, o)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1

    print(i)
    print(error_min)

    # print(list(map(lambda xj: copysign(1, xj @ w), x)))
    # print(y)
    # print(w_min)
    #
    # figures = []
    #
    # # Generate input points
    # figures.append(
    #     go.Scatter(
    #         x=x[:, 1],
    #         y=x[:, 2],
    #         mode="markers",
    #         marker=dict(
    #             size=15,
    #             color=((x[:, 1] == 1) & (x[:, 2] == 1)).astype(int),
    #             colorscale=[[0, 'red'], [1, 'black']]
    #         )
    #     )
    # )
    #
    # # Generate hyperplane
    # x_vals = arange(-1, 2)
    # figures.append(
    #     go.Scatter(
    #         x=x_vals,
    #         y=(-w_min[1] / w_min[2]) * x_vals - w_min[0] / w_min[2],
    #     )
    # )
    #
    # fig = go.Figure(
    #     figures
    #     ,
    #     {
    #         'title': f'And'
    #     }
    # )
    # fig.show()


def calculate_error(
        y: NDArray[float],
        o: NDArray[float],
):
    error = 0
    for i in range(len(y)):
        if not math.isclose(y[i], o[i], abs_tol=0.1):
            error += 1

    return error
