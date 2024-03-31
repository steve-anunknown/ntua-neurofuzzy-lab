import numpy as np
from optimizers import gradient_descent, momentum_method

def f(x1, x2, A=10):
    return A*x1**2 + (1/A)*x2**2

def df(x1, x2, A=10):
    return np.array([2*A*x1, (2/A)*x2])

def ddf(x1, x2, A=10):
    return 2*A + (2/A)

def hessian(x1, x2, A=10):
    return np.array([
        2*A, 0,
        0, 2/A
        ])

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    As = [1.2*i for i in range(1, 11)]
    initial = np.array([50, -100])
    print("========= ========= gradient descent ========= =========")
    for A in As:
        minimum, minimizer, loops = gradient_descent(lambda x1, x2: f(x1, x2, A),
                                                     lambda x1, x2: df(x1, x2, A),
                                                     initial, 0.05)
        cond = max(2*A, 2/A)/min(2*A, 2/A)
        print(f"A = {A:0,.3f}, Cond num = {cond:0,.3f}: {minimum:0,.3f}, {minimizer}, {loops}")
    print("========= ========= momentum method  ========= =========")
    for A in As:
        minimum, minimizer, loops = momentum_method(lambda x1, x2: f(x1, x2, A),
                                                    lambda x1, x2: df(x1, x2, A),
                                                    initial, 0.5, 0.05)
        cond = max(2*A, 2/A)/min(2*A, 2/A)
        print(f"A = {A:0,.3f}, Cond num = {cond:0,.3f}: {minimum:0,.3f}, {minimizer}, {loops}")
