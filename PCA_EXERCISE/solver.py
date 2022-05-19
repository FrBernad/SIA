import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

    print("Eigenvectors")
    print(pca.components_)

    print("\n\nPrincipal Components")
    print(principal_components)

    print("\n\nEigenvector - First Component")
    print(pd.DataFrame(data=pca.components_[:, 0], index=df.columns.values).T)

    print("\n\nPrincipal Components - First Component")
    print(pd.DataFrame(data=principal_components[:, 0], index=df.index.values).T)
