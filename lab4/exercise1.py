import numpy as np
np.set_printoptions(precision=4, suppress=True)


def markov_chain(a, b, c):
    chain = np.array([
        [0, 1 - a - b, a, b, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1 - c, c],
        [1, 0, 0, 0, 0],
        [0, 0, 0.5, 0, 0.5]
    ])

    for transition_probabilities in chain:
        assert sum(transition_probabilities) == 1
    return chain


if __name__ == '__main__':
    # c can be anything from 0 to 1
    # a and b must be such that a + b = 1
    c = 1/3
    a = 1/4
    b = 1 - a

    chain = markov_chain(a, b, c)
    num_states = len(chain)
    names = [f"q{i+1}" for i in range(num_states)]

    # 1: try a variety of values a, b, c
    # and find the recursive class and periodicity
    # of the markov chain.
    message = "Part A of the exercise"
    print('=' * len(message))
    print(message)
    print('=' * len(message))
    for a, b, c in [(0, 0.1, 0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2)]:
        print(f"Values a = {a}, b = {b}, c = {c}")
        chain = markov_chain(a, b, c)

        eigvals, eigvecs = np.linalg.eig(chain.T)
        left_eigvec = eigvecs[:, np.isclose(eigvals, 1)]
        left_eigvec = left_eigvec / left_eigvec.sum()
        for state, prob in zip(names, left_eigvec):
            print(f"\tState {state}: {np.abs(prob[0]):.3f}")

    # 2: if a = c = 0 and b = 0.1, what
    # is the probability that the state at t = 1000,
    # ..., 1003 is 1, ..., 5 as a function of the
    # original state.
    message = "Part B of the exercise"
    print('=' * len(message))
    print(message)
    print('=' * len(message))
    for a, b, c in [(0, 0.1, 0), (0.1, 0.1, 0.1)]:
        print(f"Values a = {a}, b = {b}, c = {c}")
        chain = markov_chain(a, b, c)

        time_steps = [1000, 1001, 1002, 1003]
        long_chain = [np.linalg.matrix_power(chain, t) for t in time_steps]
        for starting in range(num_states):
            print(f"\tStarting at state {names[starting]}")
            initial_conditions = [0] * num_states
            initial_conditions[starting] = 1
            for time in range(len(time_steps)):
                probabilities = np.dot(initial_conditions, long_chain[time])
                print(f"\t\ttime {time_steps[time]} -> probs: {probabilities}")

    # 3: for a = b = c = 0.1, simulate a sample path of the Markov chain.
    # How much time X is spent on each state? Compare with the left eigenvector
    # of the transition matrix.
    message = "Part C of the exercise"
    print('=' * len(message))
    print(message)
    print('=' * len(message))
    a = 0.1
    b = 0.1
    c = 0.1
    chain = markov_chain(a, b, c)
    total_time = 10000
    times = np.zeros(num_states)
    state = 0  # np.random.randint(num_states)
    for _ in range(total_time):
        state = np.random.choice(num_states, p=chain[state])
        times[state] += 1
    print(f"Time spent on each state out of {total_time} time steps")
    for state, time in zip(names, times):
        print(f"\tState {state}: {time}")

    print("Left eigenvector of the transition matrix")
    eigvals, eigvecs = np.linalg.eig(chain.T)
    left_eigvec = eigvecs[:, np.isclose(eigvals, 1)]
    left_eigvec = left_eigvec / left_eigvec.sum()
    for state, prob in zip(names, left_eigvec):
        print(f"\tState {state}: {np.abs(prob[0]):.3f}")
