import plotly.graph_objects as go
from numpy import matmul, array, square
from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from algorithms.oja import Oja
from utils.config import get_config


def box_plot(df, title):
    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
              'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)', 'rgba(100, 80, 200, 0.5)']
    fig = go.Figure()
    for column, color in zip(df.columns.values, colors):
        fig.add_trace(go.Box(
            y=df[column].values,
            fillcolor=color,
            name=column,
            boxmean='sd'
        ))
    fig.update_layout(
        title=f'Europe countries metrics {title}',
        margin=dict(
            l=40,
            r=30,
            b=80,
            t=100,
        ),
        showlegend=False
    )
    fig.show()


if __name__ == "__main__":
    config = get_config("../config.yaml")
    config.input_file = "../data/europe.csv"

    non_standardized_df = read_csv(config.input_file)
    non_standardized_df.index = non_standardized_df.Country.values
    non_standardized_df.drop('Country', axis=1, inplace=True)

    # box_plot(non_standardized_df, 'denormalized')

    standardized_df = DataFrame(StandardScaler().fit_transform(non_standardized_df.values))
    standardized_df.index = non_standardized_df.index
    standardized_df.columns = non_standardized_df.columns

    # box_plot(standardized_df, 'normalized')

    input_values = standardized_df.values
    oja_network = Oja(input_values, config.oja)

    result = oja_network.train()

    pca = PCA()
    principal_components = pca.fit_transform(standardized_df.values)

    # 1 EIG LOADINGS
    eig = pca.components_[0]
    fig = go.Figure(
        data=go.Bar(
            x=eig,
            y=standardized_df.columns,
            orientation='h'),
        layout=go.Layout(
            title="Loadings PCA 1 - First Eig"
        )
    )
    fig.show()

    # APPROX 1 EIG LOADINGS
    approximated_eig = result.w[-1]
    fig = go.Figure(
        data=go.Bar(
            x=approximated_eig,
            y=standardized_df.columns,
            orientation='h'),
        layout=go.Layout(
            title="Approximated Loadings PCA 1 - Approximated First Eig"
        )
    )
    fig.show()

    go.Figure(
        data=go.Table(
            header=dict(values=["", *standardized_df.columns.values],
                        fill_color='paleturquoise',
                        align='left'
                        ),
            cells=dict(values=[['EIG 1', 'EIG 1 Approx'], *array([eig, approximated_eig]).T],
                       fill_color='lavender',
                       align='left',
                       format=[None, ".4f"]
                       )
        ),
        layout=go.Layout(
            title="First Eig"
        )
    ).show()

    # PCA 1
    pca_1 = principal_components[:, 0]
    fig = go.Figure(
        data=go.Bar(
            x=pca_1,
            y=standardized_df.index,
            orientation='h'),
        layout=go.Layout(
            title="PCA 1"
        )
    )
    fig.show()

    # APPROX PCA 1
    approximated_pca = matmul(input_values, result.w[-1])
    go.Figure(
        data=go.Bar(
            x=approximated_pca,
            y=standardized_df.index,
            orientation='h'),
        layout=go.Layout(
            title="Approximated PCA 1"
        )
    ).show()

    go.Figure(
        data=go.Table(
            header=dict(values=["", 'PCA 1', 'PCA 1 Approx'],
                        fill_color='paleturquoise',
                        align='left'
                        ),
            cells=dict(values=[standardized_df.index.values, *array([pca_1, approximated_pca])],
                       fill_color='lavender',
                       align='left',
                       format=[None, ".4f"]
                       )
        ),
        layout=go.Layout(
            title="PCA 1"
        )
    ).show()

    # PCA 2
    pca_2 = principal_components[:, 1]
    fig = go.Figure(
        data=go.Bar(
            x=pca_2,
            y=standardized_df.index,
            orientation='h'),
        layout=go.Layout(
            title="PCA 2"
        )
    )
    fig.show()

    lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    error = []
    for lr in lrs:
        config.oja.learning_rate = lr
        oja_network = Oja(input_values, config.oja)
        result = oja_network.train()
        eig = pca.components_[0]
        approximated_eig = result.w[-1]
        # if eig[0] > 0 and approximated_eig[0] < 0:
        #     approximated_eig = approximated_eig * -1
        error.append((square(abs(eig) - abs(approximated_eig))).mean(axis=0))

    go.Figure(
        data=go.Bar(
            x=list(map(lambda lr: "{0:f}".format(lr), lrs)),
            y=error
        )
        ,
        layout=go.Layout(
            title="MSE per learning rate"
        )
    ).show()
