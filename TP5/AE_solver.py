import sys

from algorithms.autoencoder import Autoencoder
from data.fonts import FONTS
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_font


def AE_solver(config_file: str):
    print('--- WELCOME TO THE AE SOLVER ---')

    print(f'\tparsing config file...')
    config = get_config(config_file)

    if config.font is None:
        config.font = 2

    if config.selection_amount is None or config.selection_amount > len(FONTS[0].get('array')[0]):
        config.selection_amount = len(FONTS[0].get('array')[0])

    print(f'\tparsing font file: {config.font}')
    font = parse_font(config.font, config.selection_amount)

    print(f'\tGenerating AE Network...')
    autoencoder = Autoencoder(font.get('array'), font.get('array'), config)

    print(f'\tTraining network...')
    result = autoencoder.train()

    print(f'\tFinished!')


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:], 'AE')

    config_file = arguments['config_file']

    try:
        AE_solver(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
