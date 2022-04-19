import plotly.graph_objects as go
from numpy import random
from numpy import zeros, copysign, array, arange
from numpy.typing import NDArray


def simple_perceptron(
        dimension: int,
        p: int,
        x: NDArray[[float, float, float]],
        y: NDArray[float],
        u: float,
        threshold: int
):
    i = 0
    w = zeros(dimension + 1)
    w_min = w
    error = 1
    error_min = p * 2

    while error > 0 and i < threshold:
        i_x = random.randint(0, p)
        h = x[i_x] @ w
        o = copysign(1, h)
        delta_w = u * (y[i_x] - o) * x[i_x]
        w += delta_w
        error = calculate_error(x, y, w, p)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
    print(i)

    print(list(map(lambda xj: copysign(1, xj @ w), x)))
    print(y)
    print(w_min)

    figures = []

    # Generate input points
    figures.append(
        go.Scatter(
            x=x[:, 1],
            y=x[:, 2],
            mode="markers",
            marker=dict(
                size=15,
                color=((x[:, 1] == 1) & (x[:, 2] == 1)).astype(int),
                colorscale=[[0, 'red'], [1, 'black']]
            )
        )
    )

    # Generate hyperplane
    x_vals = arange(-1, 2)
    figures.append(
        go.Scatter(
            x=x_vals,
            y=(-w_min[1] / w_min[2]) * x_vals - w_min[0] / w_min[2],
        )
    )

    fig = go.Figure(
        figures
        ,
        {
            'title': f'And'
        }
    )
    fig.show()


def calculate_error(
        x: NDArray[[float, float, float]],
        y: NDArray[float],
        w: NDArray[float],
        p: int
):
    o = []
    for i in range(p):
        o.append(abs(y[i] - copysign(1, x[i] @ w)))

    return sum(o)


if __name__ == "__main__":
    x = array([[1, -1, 1], [1, 1, -1], [1, -1, -1], [1, 1, 1]])
    y = array([-1, -1, -1, 1])

    simple_perceptron(2, len(x), x, y, 0.01, 10000)
