import sys

import numpy as np
import pandas as pd
from numpy import matmul
from sklearn.decomposition import PCA

from algorithms.oja import Oja
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_input_csv


def oja_solver(config_file: str):
    np.set_printoptions(suppress=True, linewidth=np.inf)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    print('--- WELCOME TO THE OJA SOLVER ---')

    print(f'\tparsing config file...')
    config = get_config(config_file)

    if config.input_file is None:
        config.input_file = 'data/europe.csv'

    print(f'\tparsing input file: {config.input_file}')
    input_values = parse_input_csv(config.input_file)

    print(f'\tGenerating Kohonen Network...')
    oja_network = Oja(input_values, config.oja)

    print(f'\tTraining network...')
    result = oja_network.train()

    print(f'\tFinished!')

    print("\n\nApproximated Eigenvector - First Component")
    print(result.w[-1])

    print("\n\nApproximated PCA 1")
    print(matmul(input_values, result.w[-1]))

    pca = PCA()
    principal_components = pca.fit_transform(oja_network.input_values)
    print("\n\nEigenvector - First Component")

    print(pca.components_.T[:, 0])

    print("\n\nPCA 1")
    print(principal_components[:, 0])


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:], 'OJA')

    config_file = arguments['config_file']

    try:
        oja_solver(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
