from optimizers import gradient_descent
from numdifftools import Gradient
import numpy as np

def f(x1, x2):
    return max(x1 + x2,
               0.9*x1 - 1.1*x2 + 1,
               -0.8*x1 + 1.2*x2 -1,
               2 - 1.1*x1 - 0.9*x2)

def df(x1, x2):
    return Gradient(f)(x1, x2)

if __name__ == '__main__':
    step = 0.01 # the optimization oscillates if the step is not small enough
    # this can be attributed to the fact that the function under optimization
    # is not smooth.
    for _ in range(20):
        guess = np.array([np.random.uniform(5), np.random.uniform(5)])
        minimum, minimizer, loops = gradient_descent(f, df, guess, step, limit=10000)
        with np.printoptions(precision=3, suppress=True):
            print('%.3f' % minimum, minimizer, loops, guess)
