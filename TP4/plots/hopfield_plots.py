from itertools import combinations
from typing import List
from plotly.subplots import make_subplots

import plotly.graph_objects as go

from numpy import array, matmul, tril, fill_diagonal, sort, average, absolute, flip, reshape
from numpy.typing import NDArray

from utils.config import get_config
from utils.parser_utils import parse_letters


def _calculate_orthogonality(input_values) -> List:
    values = []
    for comb in combinations(input_values.keys(), 4):
        number_comb = array([input_values.get(key) for key in comb])
        orthogonality = tril(matmul(number_comb, number_comb.T))
        fill_diagonal(orthogonality, 0)
        orthogonality = absolute(orthogonality[orthogonality != 0])
        avg = average(orthogonality)
        max_val = max(orthogonality)
        values.append(dict(combination=' '.join(comb), avg=avg, max_value=max_val))

    return sorted(values, key=lambda i: (i['avg'], i['max_value']))


def most_orthogonal_chart(values: List):
    vals = array(list(map(lambda v: [v['combination'], v['avg'], v['max_value']], values[:20])))
    fig = go.Figure(
        data=go.Table(
            columnwidth=[1, 1.5],
            header=dict(
                values=["Combination", "Orthogonality", "Max Orthogonality"],
                align=['center', 'center', 'center']
            ),
            cells=dict(
                values=array(vals).T,
                align=['center', 'center', 'center'],
                font_size=12,
                format=[None, ".4f", None],
                height=30
            ),
        )
    )

    fig.show()


def print_letters(input_values):
    fig = make_subplots(
        rows=5, cols=6,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    row = 1
    col = 1
    for val in input_values.values():
        if col == 7:
            col = 1
            row += 1

        fig.add_heatmap(
            z=flip(reshape(val, (5, 5)), axis=0),
            showscale=False,
            row=row, col=col, colorscale='Greys',
        )
        col += 1

    fig.show()


if __name__ == "__main__":
    config = get_config("../config.yaml")
    config.input_file = "../data/font.txt"

    input_values = parse_letters(config.input_file)

    orthogonal_values = _calculate_orthogonality(input_values)

    most_orthogonal_chart(orthogonal_values)

    print_letters(input_values)
