import getopt
import sys
from typing import List

DEFAULT_CONFIG_FILE = './config.yaml'


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
    if exercise == 'AE':
        print(
            f'''NAME
\tLinear Autoencoder solver - solves font data recognition using AE\n
SYNOPSIS
\tAE_solver.py [-h] [-c config_file]\n
DESCRIPTION
\tFont data recognition using AE.
\tMandatory arguments for short options are.\n
\t-h
\t\tShow this help message and exit\n
\t-c config_file
\t\tSpecifies the .yaml configuration file. Defaults to config.yaml.\n
         ''')
    elif exercise == 'DAE':
        print(
            f'''NAME
\tDenoising Autoencoder Solver - solves font denoising using DAE\n
SYNOPSIS
\tDATE_solver.py [-h] [-c config_file]\n
DESCRIPTION
\tFont denoising using DAE.
\tMandatory arguments for short options are.\n
\t-h
\t\tShow this help message and exit\n
\t-c config_file
\t\tSpecifies the .yaml configuration file. Defaults to config.yaml.\n
         ''')
    elif exercise == 'VAE':
        print(
            f'''NAME
\tVariational Autoencoder Solver - new samples generation using VAE\n
SYNOPSIS
\tVAE_solver.py [-h] [-c config_file]\n
DESCRIPTION
\tNew samples generation using VAE.
\tMandatory arguments for short options are.\n
\t-h
\t\tShow this help message and exit\n
\t-c config_file
\t\tSpecifies the .yaml configuration file. Defaults to config.yaml.\n
         ''')
