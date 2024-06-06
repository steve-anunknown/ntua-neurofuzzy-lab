import random
import numpy as np
from optimizers import gradient_descent

def f(x1, x2, alphas, betas):
    return 1/4*(sum((x1-a)**2 + (x2-b)**2 for a, b in zip(alphas, betas)))

def df(x1, x2, alphas, betas):
    return 1/2 * np.array([sum(x1-a for a in alphas),
                           sum(x2-b for b in betas)])

def df_sample(x1, x2, alphas, betas):
    a, b = random.choice(list(zip(alphas, betas)))
    return 1/2 * np.array([x1 - a, x2 - b])

# the expected value of df_samples is equal
# to the gradient of f, therefore stochastic
# gradient descent can be used.

# instead of passing df as a parameter, we just
# pass the sampling function.

if __name__ == '__main__':
    step = 0.1
    alphas = [random.randint(0, 10) for _ in range(4)]
    betas = [random.randint(0, 10) for _ in range(4)]
    loops_stoch, loops_ord = [], []
    print("minimum | minimizer | iterations | initial guess | method")
    for _ in range(10):
        guess = np.array([np.random.uniform(10), np.random.uniform(10)])
        minimum, minimizer, loops = gradient_descent(
                lambda x1, x2: f(x1, x2, alphas, betas),
                lambda x1, x2: df_sample(x1, x2, alphas, betas),
                guess, step)
        loops_stoch.append(loops)
        with np.printoptions(precision=3, suppress=True):
           print('%.3f' % minimum, '|', minimizer, '|', loops, '\t|', guess, '|', "stochastic")
        minimum, minimizer, loops = gradient_descent(
                lambda x1, x2: f(x1, x2, alphas, betas),
                lambda x1, x2: df(x1, x2, alphas, betas),
                guess, step)
        loops_ord.append(loops)
        with np.printoptions(precision=3, suppress=True):
           print('%.3f' % minimum, '|', minimizer, '|', loops, '\t|', guess, '|', "ordinary")
    # calculate the harmonic mean of the number of iterations
    # for both methods
    print("harmonic mean of iterations for stochastic gradient descent:", 10/sum(1/i for i in loops_stoch))
    print("harmonic mean of iterations for ordinary gradient descent:", 10/sum(1/i for i in loops_ord))
