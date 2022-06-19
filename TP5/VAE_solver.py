import sys

from algorithms.VAE import VAE, flatten_set
from algorithms.autoencoder import Autoencoder
from data.fonts import FONTS
from plots.AE_font_plot import print_letters, plot_latent_layer
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_font


def VAE_solver(config_file: str):
    print('--- WELCOME TO THE VAE SOLVER ---')

    print(f'\tparsing config file...')
    config = get_config(config_file)

    if config.font is None:
        config.font = 2

    if config.selection_amount is None or config.selection_amount > len(FONTS[0].get('array')[0]):
        config.selection_amount = len(FONTS[0].get('array')[0])

    print(f'\tparsing font file: {config.font}')
    font = parse_font(config.font, config.selection_amount)

    var_autoencoder = VAE(font.get('array'), font.get('array'), config)
    print(f'\tTraining network...')
    print_letters(font.get('array'))
    var_autoencoder.train(font.get('array'), config.max_iter, len(config.intermediate_layers))
    encoder_imgs = var_autoencoder.encoder.predict(font.get('array'), batch_size=len(font.get('array')[0]))
    decoder_imgs = var_autoencoder.decoder.predict(encoder_imgs, batch_size=len(font.get('array')[0]))
    print_letters(decoder_imgs)
    print(f'\tFinished!')


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:], 'AE')

    config_file = arguments['config_file']

    try:
        VAE_solver(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
