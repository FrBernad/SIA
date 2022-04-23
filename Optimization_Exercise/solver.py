import time
from math import exp
from numpy import array, inf, zeros
from numpy.typing import NDArray
from autograd.misc.optimizers import adam
from scipy.optimize import minimize
import numdifftools as nd

xi = array(
    [
        [4.4793, -4.0765, -4.0765],
        [-4.1793, -4.9218, 1.7664],
        [-3.9429, -0.7689, 4.8830]
    ]
)

zeta = array([0., 1., 1.])


def g(
        x: float
):
    return exp(x) / (1 + exp(x))


def F(
        W: NDArray,
        w: NDArray,
        w0: NDArray,
        xi: NDArray
):
    outer_sum = 0
    for j in range(0, 2):
        inner_sum = 0
        for k in range(0, 3):
            inner_sum += w[j, k] * xi[k]
        inner_sum -= w0[j]
        outer_sum += W[j + 1] * g(inner_sum)

    return g(outer_sum - W[0])


def E(
        x: NDArray,
        step=2
):
    error = 0
    W = array(x[0:3])
    w = array([x[3:6], x[6:9]])
    w0 = array(x[9:11])

    for u in range(0, 3):
        error += (zeta[u] - F(W, w, w0, xi[u])) ** 2

    return error


def main():
    x = zeros(11)

    start_time = time.time()
    CG = minimize(E, x, method="CG",
                  options={'gtol': 1e-05, 'norm': inf, 'eps': 1.4901161193847656e-08, 'maxiter': None,
                           'disp': False, 'return_all': False, 'finite_diff_rel_step': None}
                  )

    print(f'''
GRADIENTE CONJUGADO\n
    W = {CG.x[0:3]}
    w = {CG.x[3:6]}\n\t\t{CG.x[6:9]}
    w0 = {CG.x[9:11]}
    Error = {CG.fun}
    Time = {time.time() - start_time}
    ''')

    start_time = time.time()
    BFGS = minimize(E, x, method='L-BFGS-B',
                    options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08,
                             'maxfun': 15000, 'maxiter': 15000,
                             'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})

    print(f'''
GRADIENTE DESCENDIENTE\n
    W = {BFGS.x[0:3]}
    w = {BFGS.x[3:6]}\n\t\t{BFGS.x[6:9]}
    w0 = {BFGS.x[9:11]}
    Error = {BFGS.fun}
    Time = {time.time() - start_time}
    ''')

    start_time = time.time()
    ADAM = adam(nd.Gradient(E), x, step_size=0.01)

    print(f'''
ADAM\n
    W = {ADAM[0:3]}
    w = {ADAM[3:6]}\n\t\t{ADAM[6:9]}
    w0 = {ADAM[9:11]}
    Error = {E(ADAM)}
    Time = {time.time() - start_time}
    ''')


if __name__ == "__main__":
    main()
