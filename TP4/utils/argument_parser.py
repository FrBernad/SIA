import getopt
import sys
from typing import List

DEFAULT_CONFIG_FILE = 'config.yaml'


def parse_arguments(argv: List[str], exercise: str) -> dict:
    arguments = {
        'config_file': DEFAULT_CONFIG_FILE,
    }
    try:
        opts, args = getopt.getopt(argv, "hc:")
    except getopt.GetoptError:
        _help(exercise)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            _help(exercise)
            sys.exit()
        elif opt in "-c":
            arguments['config_file'] = arg

    return arguments


def _help(exercise: str):
    if exercise == 'KOHONEN':
        print(
            f'''NAME
\tKohonen Solver - solves data grouping problem using Self-Organizing Map\n
SYNOPSIS
\tkohonen_solver.py [-h] [-c config_file]\n
DESCRIPTION
\tSolves data grouping problem using Self-Organizing Map for Europe countries.
\tMandatory arguments for short options are.\n
\t-h
\t\tShow this help message and exit\n
\t-c config_file
\t\tSpecifies the .yaml configuration file. Defaults to config.yaml.\n
         ''')
