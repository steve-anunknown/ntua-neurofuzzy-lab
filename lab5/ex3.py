import random

def control(u, index, num_states):
    if not (u == -1 or u == 1):
        raise ValueError("u must be either -1 or 1")

    flag = random.choice([False, True])
    if not flag:
        return index
    if u == 1:
        return min(index + 1, num_states - 1)
    elif u == -1:
        return max(index - 1, 0)

def law(index):
    return random.choice([-1, 1])

if __name__ == '__main__':

    # this array of values is the cost
    # per state
    cost = [1, 2, 3, 4, 5, 4, 2, 0, 1, 2]

    num_states = 10
    states = range(num_states)

    # A. Suppose that the controller is known.
    # Find the trajectories of the system

    # B. Implement Q-learning for this system.
    # Test the effect of different parameters
    # on the convergence speed.


    
