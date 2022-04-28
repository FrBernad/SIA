from numpy import array
from numpy.typing import NDArray


def parse_training_values(file: str) -> NDArray:
    with open(file) as df:
        lines = df.readlines()
        values = []

        for entry in lines:
            # float_vals = [1, *list(map(lambda v: float(v), entry.split()))]
            float_vals = [*list(map(lambda v: float(v), entry.split())), 1]
            values.append(float_vals)

        return array(values)


def parse_output_values(file: str) -> NDArray:
    with open(file) as df:
        lines = df.readlines()
        values = []

        for entry in lines:
            float_vals = list(map(lambda v: float(v), entry.split()))
            values.append(float_vals)

        return array(values)


def parse_nums(file: str) -> NDArray:
    with open(file) as df:
        lines = df.readlines()
        values = []

        for entry in lines:
            float_vals = list(map(lambda v: float(v), entry.split()))[0]
            values.append(float_vals)

        return array(values)
