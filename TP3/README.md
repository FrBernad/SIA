# Authors

- [Francisco Bernad](https://github.com/FrBernad)
- [Nicolás Rampoldi](https://github.com/NicolasRampoldi)
- [Joaquín Legammare](https://github.com/JoacoLega)

# NEURAL NETWORKS OPTIMIZATIONS

This project aims to solve different problems using different neural networks strategies.
It makes use of:

- Simple Perceptron
- Linear Perceptron
- Non-Linear Perceptron
- Multilayer Perceptron

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
  python ejN.py [-h] [-c config_file]
```

Where:

- config_file: specifies the path to a valid yaml configuration file. Defaults to **"config.yaml"**
- data_file: specifies the .txt data file to generate the knapsack. Defaults to  **"Mochila100Elementos.txt"**.
- output_file: specifies the path to the output file containing the found solution. Defaults to **"solution.yaml"**

## Configuration File

The configuration file must be a valid yaml file with the following structure:

```
# --- Neural Network Config ---
config:

  # Input and Output file paths for the training values
  training_values:
    input: training_values/ej3-2-input.txt
    output: training_values/ej3-2-output.txt
    #    output: training_values/ej3-3-output.txt
  #    input: training_values/ej1-xor-input.txt
  #    output: training_values/ej1-xor-output.txt
  #    input: training_values/ej2-linear-input.txt
  #    output: training_values/ej2-linear-output.txt
  #      input:
  #      output:

  # Plot results
  plot: true

  #Perceptron config
  perceptron:
    #Perceptron type
    #Values:
    # - simple
    # - linear
    # - non_linear
    # - multilayer
    type: linear

    #Perceptron specific settings
    settings:
      #Global
      learning_rate: 0.1
      min_error: 0.00001
      min_iter: 5000

      #Non-Linear
      g: tanh
      b: 0.4
      
      #Momentum and adaptative lr
      variable: false
      alpha: 0.8
      a: 0.01
      beta: 0.1

...
```