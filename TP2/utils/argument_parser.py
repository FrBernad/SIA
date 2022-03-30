import getopt
import sys
from typing import List

DEFAULT_CONFIG_FILE = 'config.yaml'
DEFAULT_DATA_FILE = 'Mochila100Elementos.txt'
DEFAULT_OUTPUT_FILE = 'solution.yaml'

def parse_arguments(argv: List[str]) -> dict:
    arguments = {
        'config_file': DEFAULT_CONFIG_FILE,
        'data_file': DEFAULT_DATA_FILE,
        'output_file': DEFAULT_OUTPUT_FILE,
    }
    try:
        opts, args = getopt.getopt(argv, "hc:d:o:")
    except getopt.GetoptError:
        _help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            _help()
            sys.exit()
        elif opt in "-c":
            arguments['config_file'] = arg
        elif opt in "-d":
            arguments['data_file'] = arg
        elif opt in "-o":
            arguments['output_file'] = arg

    return arguments


def _help():
    print(
f'''NAME
\tknapsack_solver - solves the 0-1 knapsack problem\n
SYNOPSIS
\tknapsack_solver.py [-h] [-c config_file] [-d data_file] [-o output_file]\n
DESCRIPTION
\tSolves the 0-1 knapsack problem using genetic algorithms.
\tMandatory arguments for short options are.\n
\t-h
\t\tShow this help message and exit\n
\t-c config_file
\t\tSpecifies the .yaml configuration file. Defaults to config.yaml.\n
\t-d data_file
\t\tSpecifies the .txt data file to generate the knapsack. Defaults to Mochila100Elementos.txt.\n
\t-o output_file
\t\tSpecifies the .yaml output file containing the solution. Defaults to solution.yaml.\n
         ''')
