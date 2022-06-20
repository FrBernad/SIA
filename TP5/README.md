# Authors

- [Francisco Bernad](https://github.com/FrBernad)
- [Nicolás Rampoldi](https://github.com/NicolasRampoldi)
- [Joaquín Legammare](https://github.com/JoacoLega)

# AUTOENCODERS NETWORKS

This project aims to solve different problems using a variety of autoencoders.
It makes use of:
- Linear Auto Enconder
- Denoising Autoencoder
- Variational Autoencoders

# Requirements

- [Python 3+](https://www.python.org/downloads/)
- [Pipenv](https://pipenv.pypa.io/en/latest/)

# Setup

On the root project folder run the command `pipenv install`. This will install all necessary Python dependencies get the
project up and running.

# Usage

To run the different exercises you **MUST** follow the next steps:

1. Generate or modify a yaml config file specifying the parameters to be used. You can find a default
   config file (***config.yaml***) explaining all possible parameters value.
2. Run the solver with the following command.

```bash
  pipenv shell 
  python [encoder]_solver.py [-h] [-c config_file]
```

Where:

- config_file: specifies the path to a valid yaml configuration file. Defaults to **"config.yaml"**

## Configuration File

The configuration file must be a valid yaml file with the following structure:

```
---
# --- Autoencoders Config ---
config:

  # Input and Output file paths for the input values
  # Values:
  # - 1 numbers
  # - 2 uppercase characters
  # - 3 lowercase characters
  font: 2

  # font values selection amount
  selection_amount: 5

  #network max iterations
  max_iter: 100
  #network min error
  min_error: 0.01
  #network latent layer
  latent_layer: 2
  #network intermediate layers
  intermediate_layers: [25,10]
...
```
