# Authors

- [Francisco Bernad](https://github.com/FrBernad)
- [Nicolás Rampoldi](https://github.com/NicolasRampoldi)
- [Joaquín Legammare](https://github.com/JoacoLega)

# NON-LINEAR OPTIMIZATION EXERCISE SOLVER

This project aims to optimize the parameters ***W***, ***w*** and ***w0*** for the error function 
defined in the [Assignment.pdf](./Assignment.pdf) file. 
It uses three non-linear optimization methods to get the results:
- [Gradient Descent Method](https://en.wikipedia.org/wiki/Gradient_descent)
- [Conjugate Gradient Method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
- [ADAM Method](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)

# Requirements

- [Python 3+](https://www.python.org/downloads/)
- [Pipenv](https://pipenv.pypa.io/en/latest/)

# Setup

On the root project folder run the command `pipenv install`. This will install all necessary Python dependencies get the
project up and running.

# Usage

To run the optimization solver you **MUST** follow the next steps:

1. Run the solver with the following command.

```bash
  pipenv shell 
  python solver.py
```

# Output

The solver will output the following values for each method:
- Optimus values for ***W***, ***w*** and ***w0***
- Optimus error
- Time

Example output:
```
METODO GRADIENTE CONJUGADO

    W = [6.1497914  7.12182008 7.12182008]
    w = [-2.76091052  0.5392962   2.34593898]
		[-2.76091052  0.5392962   2.34593898]
    w0 = [0.0628374 0.0628374]
    Error = 4.720716971989189e-06
    Time = 0:00:00.014007
```
