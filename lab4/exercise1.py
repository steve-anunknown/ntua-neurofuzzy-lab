

if __name__ == '__main__':
    alpha = 0.1
    beta = 0.1
    gamma = 0.1

    markov_chain = 
    [
        [0, 1 - alpha - beta, alpha, beta, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1 - gamma, gamma],
        [1, 0, 0, 0, 0],
        [0, 0, 0.5, 0, 0.5]
    ]
    for transition_probabilities in markov_chain:
        assert sum(transition_probalities) == 1

    num_states = len(markov_chain)
    names = [f"q{i+1}" for i in range(num_states)]


    # 1: try a variety of values alpha, beta, gamma
    # and find the recursive class and periodicity
    # of the markov chain.

    # 2: if alpha = gamma = 0 and beta = 0.1, what
    # is the probability that the state at t = 1000,
    # ..., 1003 is 1, ..., 5 as a function of the
    # original state.
