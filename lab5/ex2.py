import matplotlib.pyplot as plt
import numpy as np
import random

def monte_carlo(n):
    count = 0
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x*x + y*y <= 1:
            count += 1
    return 4 * count / n


if __name__ == '__main__':
    iterations = 10000
    print(monte_carlo(iterations))
