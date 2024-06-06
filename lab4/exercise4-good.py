import numpy as np

np.set_printoptions(precision=3)


def compute_optimal_policy(g, num_states, discount_factor):
    # Initialize the value function array
    V = np.zeros(num_states)
    V_new = np.zeros(num_states)
    # Initialize the policy array
    policy = np.zeros(num_states, dtype=int)

    # Tolerance for convergence
    iters = 0
    while iters < 1000:
        for i in range(num_states):
            # Compute the value for both actions (+1 and -1)
            v_plus = g[i] + discount_factor * (0.5 * V[min(i + 1, num_states - 1)] + 0.5 * V[i])
            v_minus = g[i] + discount_factor * (0.5 * V[max(i - 1, 0)] + 0.5 * V[i])

            # Choose the action with the minimum value
            min_value = min(v_plus, v_minus)

            # Update the policy
            policy[i] = 1 if v_plus < v_minus else -1

            # Update the value function
            V_new[i] = min_value

            iters += 1

        V = np.copy(V_new)

    return V, policy


if __name__ == '__main__':
    # The given g function
    g = np.array([1, 2, 3, 4, 5, 4, 2, 0, 1, 2])

    num_states = 10

    # different values of a
    base = 0.2
    discount_factors = [base * (i + 1) for i in range(10)]

    for a in discount_factors:
        V, policy = compute_optimal_policy(g, num_states, a)
        print(f'Discount Factor: {a:.3f}')
        print('Optimal Policy:', policy)
        print(f'Expected Cost: {V}')
        print()

