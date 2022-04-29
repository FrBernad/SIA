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

        for i in range(10):
            number = []
            for j in range(7):
                current_line = list(map(lambda v: int(v), lines[j + i * 7].split()))
                for n in current_line:
                    number.append(n)
            number.append(1)
            values.append(number)

        return array(values)
