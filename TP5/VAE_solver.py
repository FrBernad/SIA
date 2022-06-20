import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from algorithms.VAE import VAE
from utils.argument_parser import parse_arguments
from utils.config import get_config


def VAE_solver(config_file: str):
    print('--- WELCOME TO THE VAE SOLVER ---')

    print(f'\tparsing config file...')
    config = get_config(config_file)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    autoencoder = VAE(x_train, x_train, config)

    print(f'\tTraining network...')
    autoencoder.train(x_train, x_train, len(config.intermediate_layers))

    x_test_encoded = autoencoder.encoder.predict(x_test, batch_size=100)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap='viridis')
    plt.colorbar()
    plt.show()

    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

    print(f'\tFinished!')


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:], 'VAE')

    config_file = arguments['config_file']

    # try:
    VAE_solver(config_file)
    # except FileNotFoundError as e:
    #     print("File not found")
    #     print(e)
    # except OSError:
    #     print("Error occurred.")
    # except KeyboardInterrupt:
    #     print('Program interrupted by user.')
    # except Exception as e:
    #     print(e)
