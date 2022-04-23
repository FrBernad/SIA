from typing import Callable

import plotly.graph_objects as go
from numpy import random, vectorize
from numpy import zeros, copysign, array, arange
from numpy.typing import NDArray


# FIXME: ANIMAR W
def simple_perceptron_algorithm(
        x: NDArray[[float, float, float]],
        y: NDArray[float],
        activation_function: Callable,
        delta_w_function: Callable,
        dimension: int,
        examples_count: int,
        learning_rate: float,
        threshold: int
):
    i = 0
    w = random.uniform(size=dimension + 1)
    w_min = w
    error = 1
    error_min = examples_count * 2

    while error > 0 and i < threshold:
        i_x = random.randint(0, examples_count)
        h = x @ w
        o = vectorize(activation_function)(h)

        delta_w = delta_w_function(y[i_x], o[i_x], x[i_x], learning_rate)
        w += delta_w
        error = calculate_error(y, o)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
    # print(i)
    #
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
    return sum(abs(y - o))
