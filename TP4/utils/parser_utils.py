import pandas as pd
from numpy import where, array
from numpy.typing import NDArray

from sklearn.preprocessing import StandardScaler


def parse_input_csv(input_file: str) -> NDArray:
    df = pd.read_csv(input_file)
    df.index = df.Country.values
    df.drop('Country', axis=1, inplace=True)
    return StandardScaler().fit_transform(df.values)


def parse_letters(file: str) -> dict:
    with open(file) as df:
        lines = df.readlines()
        values = []

        for i in range(26):
            number = []
            for j in range(5):
                current_line = list(map(lambda v: int(v), lines[j + i * 5].split()))
                for n in current_line:
                    number.append(n)
            values.append(number)

        values = array(values)
        values = where(values == 0, -1, values)
        letters = {}
        current_letter = 65
        for value in values:
            letters[chr(current_letter)] = value
            current_letter += 1

        return letters
