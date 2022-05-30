from itertools import combinations
from math import floor
from typing import List

import plotly.graph_objects as go
from numpy import array, matmul, tril, fill_diagonal, average, absolute, flip, reshape, copy
from numpy.random import choice
from plotly.subplots import make_subplots

from algorithms.hopfield import Hopfield
from utils.config import get_config, HopfieldConfig
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
        ),
        layout=go.Layout(
            title="Most Orthogonal Values"
        )
    )

    fig.show()


def least_orthogonal_chart(values: List):
    vals = array(list(map(lambda v: [v['combination'], v['avg'], v['max_value']], values[-20:])))
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
        ),
        layout=go.Layout(
            title="Least Orthogonal Values"
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


def print_combination(input_values, combination):
    numbers_array = array([input_values.get(key) for key in combination['combination'].split()])

    fig = make_subplots(
        rows=2, cols=len(numbers_array),
        subplot_titles=[f"{combination['combination']} - Orthogonality {combination['avg']}", ""]
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    col = 1
    for val in numbers_array:
        if col == len(numbers_array) + 1:
            col = 1

        fig.add_heatmap(
            z=flip(reshape(val, (5, 5)), axis=0),
            showscale=False,
            row=1, col=col, colorscale='Greys',
        )
        col += 1

    fig.show()


def print_patterns(input_values, mutations):
    fig = make_subplots(
        rows=2, cols=len(input_values),
        subplot_titles=[f"mutations - {mutations}", ""]
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    col = 1
    for val in input_values:
        if col == len(input_values) + 1:
            col = 1

        fig.add_heatmap(
            z=flip(reshape(val, (5, 5)), axis=0),
            showscale=False,
            row=1, col=col, colorscale='Greys',
        )
        col += 1

    fig.show()


def _mutate_input_value(input_value, mutated_amount):
    mutated_input = copy(input_value)
    indexes = choice(range(len(input_value) - 1), size=mutated_amount, replace=False)
    for i in indexes:
        mutated_input[i] = 1 if mutated_input[i] == -1 else -1

    return mutated_input


def low_mutations(config: HopfieldConfig, orthogonal_values, input_values):
    combinations = [orthogonal_values[0], orthogonal_values[floor(len(orthogonal_values) / 2)], orthogonal_values[-1]]

    for mutations in [2, 5]:
        for combination in combinations:
            numbers_array = array([input_values.get(key) for key in combination['combination'].split()])
            print_combination(input_values, combination)

            hopfield_network = Hopfield(config, numbers_array)
            for pattern in list(combination['combination'].split()):
                mutated_array = _mutate_input_value(input_values[pattern], mutations)
                results = hopfield_network.train(mutated_array)
                print_patterns([input_values[pattern], mutated_array, *results.patterns], mutations)
                print_energy_function(results)


def spurious_state(config: HopfieldConfig, orthogonal_values, input_values):
    combination = orthogonal_values[0]

    numbers_array = array([input_values.get(key) for key in combination['combination'].split()])
    print_combination(input_values, combination)

    hopfield_network = Hopfield(config, numbers_array)
    for pattern in list(combination['combination'].split()):
        mutated_array = _mutate_input_value(input_values[pattern], 15)
        results = hopfield_network.train(mutated_array)
        print_patterns([input_values[pattern], mutated_array, *results.patterns], 15)
        print_energy_function(results)


def print_energy_function(results):
    fig = go.Figure(
        data=
        go.Scatter(
            y=results.H,
            x=list(range(len(results.H)))
        ),
        layout=go.Layout(
            title="Energy function",
            xaxis=dict(title="iteration"),
            yaxis=dict(title="energy")
        )
    )
    fig.show()


if __name__ == "__main__":
    config = get_config("../config.yaml")
    config.input_file = "../data/font.txt"
    config.hopfield.max_iter = 8

    input_values = parse_letters(config.input_file)

    orthogonal_values = _calculate_orthogonality(input_values)

    # most_orthogonal_chart(orthogonal_values)
    # least_orthogonal_chart(orthogonal_values)
    #
    # print_letters(input_values)

    low_mutations(config.hopfield, orthogonal_values, input_values)
    spurious_state(config.hopfield, orthogonal_values, input_values)
