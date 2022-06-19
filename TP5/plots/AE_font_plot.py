import plotly.graph_objects as go
from numpy import array, flip, reshape, where
from numpy.linalg import norm
from plotly.subplots import make_subplots

from algorithms.autoencoder import Autoencoder
from utils.config import get_config
from utils.parser_utils import parse_font


def print_letters(font):
    fig = make_subplots(
        rows=4, cols=8
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    row = 1
    col = 1
    for val in font:
        if col == 9:
            col = 1
            row += 1

        fig.add_heatmap(
            z=flip(reshape(val, (7, 5)), axis=0),
            showscale=False,
            row=row, col=col, colorscale='Greys',
            colorbar=dict(bordercolor="black", borderwidth=1)
        )
        col += 1

    fig.show()


def plot_latent_layer(latent_values, letters):
    fig = go.Figure(
        data=[
            go.Scatter(
                x=latent_values[:, 0],
                y=latent_values[:, 1],
                text=letters,
                mode='markers',
            )
        ],
        layout=go.Layout(
            title="Latent Space",
            xaxis=dict(title="x"),
            yaxis=dict(title="y"),
        )
    )
    for i in range(len(latent_values)):
        fig.add_annotation(
            x=latent_values[i, 0],
            y=latent_values[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=letters[i],
            font=dict(size=18)
        )
    fig.update_traces(textposition='top center')
    fig.show()


if __name__ == "__main__":
    config = get_config("../config.yaml")
    config.font = 2
    config.max_iter = 50
    config.intermediate_layers = [25]

    font = parse_font(config.font, 5)
    font_array = font.get('array')
    font_letters = font.get('letters')

    print_letters(font_array)

    for layers in [[25], [25, 15], [25, 15, 10]]:
        config.intermediate_layers = layers
        autoencoder = Autoencoder(font_array, font_array, config)

        result = autoencoder.train()

        decoded_values = []
        for val in font_array:
            decoded_values.append(autoencoder.propagate(result.weights, val))
        print_letters(decoded_values)

    # latent_values = []
    # for val in font_array:
    #     latent_values.append(autoencoder.encode(val, result.weights))
    #
    # plot_latent_layer(array(latent_values), font_letters)

    # direction = latent_values[1] - latent_values[0]
    #
    # parts = 5
    #
    # direction_latent_values = [latent_values[0]]
    # direction_font_letters = [font_letters[0]]
    #
    # proportion = direction / parts
    #
    # for i in range(1, parts):
    #     direction_latent_values.append(latent_values[0] + proportion * i)
    #     direction_font_letters.append('*')
    # direction_latent_values.append(latent_values[1])
    # direction_font_letters.append(font_letters[1])
    #
    # plot_latent_layer(array(direction_latent_values), direction_font_letters)
    #
    # direction_letters = []
    # for val in direction_latent_values:
    #     direction_letters.append(autoencoder.decode(val, result.weights))
    #
    # print_letters(direction_letters)
