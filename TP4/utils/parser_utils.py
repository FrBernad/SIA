import pandas as pd
from numpy.typing import NDArray

from sklearn.preprocessing import StandardScaler


def parse_input_values(input_file: str) -> NDArray:
    df = pd.read_csv(input_file)
    df.index = df.Country.values
    df.drop('Country', axis=1, inplace=True)
    return StandardScaler().fit_transform(df.values)
