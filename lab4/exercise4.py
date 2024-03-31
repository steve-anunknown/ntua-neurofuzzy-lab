import numpy as np

# the policy is the choice that the controller makes for each state
def compute_optimal_policy(g, num_states, discount_factor):
    # Initialize the value function array
    V = np.zeros(num_states)
    # Initialize the policy array
    policy = np.zeros(num_states, dtype=int)
    
    # Tolerance for convergence
    tolerance = 1e-5
    delta = 1
    # Iterate until convergence
    while delta >= tolerance:
        delta = 0
        for i in range(num_states):
            # Compute the value for both actions (+1 and -1)
            v_plus = g[i] + discount_factor * (0.5 * V[min(i + 1, num_states - 1)] + 0.5 * V[i])
            v_minus = g[i] + discount_factor * (0.5 * V[max(i - 1, 0)] + 0.5 * V[i])

            # Choose the action with the minimum value
            min_value = min(v_plus, v_minus)

            # Update the policy
            policy[i] = 1 if v_plus < v_minus else -1

            # Calculate the maximum change in value function
            delta = max(delta, abs(min_value - V[i]))

            # Update the value function
            V[i] = min_value

    return V, policy
   
if __name__ == '__main__':
	# The given g function
	g = np.array([1, 2, 3, 4, 5, 4, 2, 0, 1, 2])

	num_states = 10

	# different values of a
	discount_factors = [0.2, 0.5, 0.7, 0.9]
	optimal_policies = {}

	for a in discount_factors:
	    V, policy = compute_optimal_policy(g, num_states, a)
	    optimal_policies[a] = policy

	print(optimal_policies)

