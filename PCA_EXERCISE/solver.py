import pandas as pd
import numpy as np
from numpy import cumsum, array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=np.inf)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    df = pd.read_csv('europe.csv')
    df.index = df.Country.values
    df.drop('Country', axis=1, inplace=True)

    x = df.values
    standardized_x = StandardScaler().fit_transform(x)

    pca = PCA()
    principal_components = pca.fit_transform(standardized_x)

    # EIGENVECTORS
    print("Eigenvectors")
    print(pca.components_)

    # PRINCIPAL COMPONENTS
    print("\n\nPrincipal Components")
    print(principal_components)

    # PRINCIPAL COMPONENTS - VARIANCE
    print("\n\nPrincipal Components - Variance Ratio")
    vr_df = pd.DataFrame(data=array([pca.explained_variance_ratio_, cumsum(pca.explained_variance_ratio_)]),
                         columns=list(map(lambda c: f'Component {c}', range(1, len(principal_components[0]) + 1))),
                         index=['Variance', 'Accumulated Variance'])

    print(vr_df)
    go.Figure(
        data=[go.Table(
            header=dict(values=["# Component", *vr_df.columns.values],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[vr_df.index, *vr_df.values.T],
                       fill_color='lavender',
                       align='left',
                       format=[None, ".4f"]
                       ),
        )],
        layout=go.Layout(
            title="Principal Components - Variance Ratio"
        )
    ).show()

    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    ).show()

    # FIRST COMPONENT AND FIRST EIG (LOADINGS)
    print("\n\nEigenvector - First Component")
    eig_1 = pd.DataFrame(data=pca.components_[:, 0], index=df.columns.values).T
    print(eig_1)

    print("\n\nPrincipal Components - First Component")
    pca_1 = pd.DataFrame(data=principal_components[:, 0], index=df.index.values).T
    print(pca_1)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Eigenvector - First Component", "Principal Components - First Component"],
        specs=[[{'type': 'table', "colspan": 2}, None], [{'type': 'table', "colspan": 2}, None]],
    )
    fig.add_table(
        row=1,
        col=1,
        header=dict(values=["Field", *eig_1.columns.values],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=['Value', *eig_1.values.T],
                   fill_color='lavender',
                   align='left',
                   format=[None, ".4f"]
                   )
    ),
    fig.add_table(
        row=2,
        col=1,
        header=dict(values=["Country", "Value"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[pca_1.columns.values, pca_1.values[0]],
                   fill_color='lavender',
                   align='left',
                   format=[None, ".4f"]
                   )
    )
    fig.show()

    # PCA_1 VS PCA_2
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = px.scatter(principal_components, x=0, y=1, text=df.index.values, color=df.index.values)
    fig.update_traces(textposition='top center')

    for i, feature in enumerate(df.columns.values):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature
        )

    fig.update_xaxes(dict(
        title=f'PCA 1 - variance {pca.explained_variance_ratio_[0] * 100:.2f}%',
    ))

    fig.update_yaxes(dict(
        title=f'PCA 2 - variance {pca.explained_variance_ratio_[1] * 100:.2f}%'
    ))

    fig.show()
