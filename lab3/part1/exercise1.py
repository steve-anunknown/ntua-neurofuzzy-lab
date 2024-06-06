import numpy as np
from optimizers import gradient_descent, newtons_method
np.set_printoptions(precision=3)


def f(x1, x2):
    return x1**2 + (x2 - 1)**2 + (x1 - x2)**4


def df(x1, x2):
    return np.array([2*x1 + 4*(x1 - x2)**3,
                     2*(x2 - 1) - 4*(x1 - x2)**3])


def ddf(x1, x2):
    return (4 + 24*(x1 - x2)**2)


if __name__ == '__main__':
    steps = [0.1 - 0.01 * i for i in range(0, 10)]
    initial = np.array([2, 5])
    print("========= ========= gradient descent ========= =========")
    for step in steps:
        minimum, minimizer, loops = gradient_descent(f, df, initial, step)
        print(f"step = {step:0,.3f}: {minimum:0,.3f}, {minimizer}, {loops}")
    steps = [0.1 + 0.1 * i for i in range(0, 10)]
    print("========= ========= newton's method  ========= =========")
    for step in steps:
        minimum, minimizer, loops = newtons_method(f, df, ddf, initial, step)
        print(f"step = {step:0,.3f}: {minimum:0,.3f}, {minimizer}, {loops}")
