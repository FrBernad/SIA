import plotly.graph_objects as go
from numpy import zeros, array
from pandas import read_csv

from algorithms.kohonen import Kohonen
from utils.config import get_config
from utils.parser_utils import parse_input_csv


def _make_plots(kohonen_network: Kohonen):
    countries = read_csv(config.input_file).Country.values
    neuron_count = zeros(shape=(config.kohonen.k, config.kohonen.k))
    neuron_countries = [['' for _ in range(config.kohonen.k)] for _ in range(config.kohonen.k)]
    for value, country in zip(input_values, countries):
        x, y = kohonen_network.get_winner(value)
        neuron_count[x][y] += 1
        neuron_countries[x][y] += f'{country} - '

    for i in range(len(neuron_countries)):
        for j in range(len(neuron_countries[i])):
            neuron_countries[i][j] = neuron_countries[i][j].rstrip(' -')

    fig = go.Figure(
        data=go.Heatmap(
            z=neuron_count,
            text=neuron_countries,
            texttemplate="%{text}\n",
            textfont={"size": 10},
            colorscale="Blues"
        ),
        layout=go.Layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            title=f'Countries per neuron - Learning rate: {kohonen_network.init_learning_rate} - K: {kohonen_network.k}'
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()

    neuron_avg = zeros(shape=(config.kohonen.k, config.kohonen.k))
    for i in range(config.kohonen.k):
        for j in range(config.kohonen.k):
            neuron_avg[i][j] = kohonen_network.get_neighbors_avg(array([i, j]))

    fig = go.Figure(
        data=go.Heatmap(
            z=neuron_avg,
            texttemplate="%{text}\n",
            textfont={"size": 10},
            colorscale="Blues"
        ),
        layout=go.Layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            title=f"U-Matrix - Learning rate: {kohonen_network.init_learning_rate} - K: {kohonen_network.k}"
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()

    fig = go.Figure(
        data=go.Heatmap(
            z=neuron_count,
            colorscale="Blues"
        ),
        layout=go.Layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            title=f"Elements per neuron - Learning rate: {kohonen_network.init_learning_rate} - K: {kohonen_network.k}"
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()


if __name__ == "__main__":
    config = get_config("../config.yaml")
    config.input_file = "../data/europe.csv"

    input_values = parse_input_csv(config.input_file)

    learning_rates = [0.001, 0.01, 0.1]
    ks = [3, 4, 5]

    for k in ks:
        for lr in learning_rates:
            config.kohonen.k = k
            config.kohonen.learning_rate = lr
            kohonen_network = Kohonen(input_values, config.kohonen)
            kohonen_network.train()
            _make_plots(kohonen_network)
