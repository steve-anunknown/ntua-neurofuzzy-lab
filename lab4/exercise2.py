import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    right_side = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

    system = [
        [1, -1/2, 0, 0, -1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1/2, 1, -1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1/3, 1, -1/3, 0, -1/3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1/2, 1, 0, 0, -1/2, 0, 0, 0, 0, 0, 0, 0],
        [-1/2, 0, 0, 0, 1, 0, 0, -1/2, 0, 0, 0, 0, 0, 0],
        [0, 0, -1/2, 0, 0, 1, -1/2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1/3, 0, -1/3, 1, 0, 0, -1/3, 0, 0, 0, 0],
        [0, 0, 0, 0, -1/3, 0, 0, 1, -1/3, 0, -1/3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1/2, 1, 0, 0, -1/2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1/2, 0, 0, 1, -1/2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1/3, 0, -1/3, 1, -1/3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/2, 1, -1/2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1/2, 0, 0, -1/2, 1]
    ]

    free_vars = np.linalg.solve(system, right_side)
    print(f"mean absorption time from each state {free_vars}")

    markov = np.array(system)
    for i in range(len(markov)):
        for j in range(len(markov)):
            if system[i][j] == 1:
                markov[i][j] = 0
            if system[i][j] < 0:
                markov[i][j] = -system[i][j]

    # simulate the system and make a histogram of the absorption times
    states = len(markov)
    simulations = 1000
    absorption_times = []
    starting = 1
    for _ in range(simulations):
        state = starting
        steps = 0
        while state != 10:
            state = np.random.choice(range(0, states), p=markov[state-1])
            steps += 1
        absorption_times.append(steps)

    plt.hist(absorption_times, bins=range(0, max(absorption_times) + 1))
    plt.title("Histogram of absorption times")
    plt.xlabel("Absorption time")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("histogram.png")
