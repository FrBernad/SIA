# Authors

- [Francisco Bernad](https://github.com/FrBernad)
- [Nicolás Rampoldi](https://github.com/NicolasRampoldi)
- [Joaquín Legammare](https://github.com/JoacoLega)

# UNSUPERVISED LEARNING

This project aims to solve different problems using a variety unsupervised machine learning algorithms.
It makes use of:
- Kohonen Networks
- Oja Rule
- Hopfield Networks

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
  python [method]_solver.py [-h] [-c config_file]
```

Where:

- config_file: specifies the path to a valid yaml configuration file. Defaults to **"config.yaml"**

## Configuration File

The configuration file must be a valid yaml file with the following structure:

```
---
# --- Unsupervised Learning Config ---
config:

  # Input and Output file paths for the input values
  input_file:

  #Kohonen config
  kohonen:
    #network learning rate
    learning_rate: 0.01
    #network radius
    radius: 4
    #network max iterations
    max_iter: 1000
    #network k neurons
    k: 4

  #Hopfield config
  hopfield:
    #network max iterations
    max_iter: 100

  #Oja config
  oja:
    #network learning rate
    learning_rate: 0.0001
    #network max iterations
    max_iter: 10000
...
```
