import plotly.graph_objects as go
from numpy import zeros, array, empty
from pandas import read_csv

from algorithms.kohonen import Kohonen
from utils.config import get_config
from utils.parser_utils import parse_input_csv


def _make_plots(kohonen_network: Kohonen):
    df = read_csv(config.input_file)
    countries = df.Country.values
    df.drop('Country', axis=1, inplace=True)
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

    for index, c in enumerate(df.columns.values):
        values = empty((len(kohonen_network.neurons), len(kohonen_network.neurons)))
        for i in range(len(kohonen_network.neurons)):
            for j in range(len(kohonen_network.neurons)):
                values[i][j] = kohonen_network.neurons[i][j][index]

        fig = go.Figure(
            data=go.Heatmap(
                z=values,
                colorscale="Blues"
            ),
            layout=go.Layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                title=f"{c}"
            )
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        fig.show()


if __name__ == "__main__":
    config = get_config("../config.yaml")
    config.input_file = "../data/europe.csv"

    input_values = parse_input_csv(config.input_file)

    learning_rates = [00.01]
    ks = [3]
    epochs = [10000]

    for k in ks:
        for lr in learning_rates:
            for e in epochs:
                config.kohonen.k = k
                config.kohonen.learning_rate = lr
                config.kohonen.max_iter = e
                kohonen_network = Kohonen(input_values, config.kohonen)
                kohonen_network.train()
                _make_plots(kohonen_network)
