import numpy as np
import random

def simulate_trajectory(initial_state, cost, controller, alpha=0.9, steps=10):
    """
    Simulate a trajectory of the controlled Markov chain.
    
    Parameters:
    - initial_state: The starting state of the system.
    - controller: The controller.
    - alpha: Discount factor for the cost.
    - max_steps: Maximum number of steps to simulate.
    
    Returns:
    - states: List of states visited during the trajectory.
    - cost: Total discounted cost of the trajectory.
    """
    states = [initial_state]
    current_state = initial_state
    total_cost = 0
    for k in range(steps):
        control = controller[current_state-1]
        if control == +1:
            next_state = current_state + 1 if np.random.rand() < 0.5 else current_state
        elif control == -1:
            next_state = current_state - 1 if np.random.rand() < 0.5 else current_state
        else:
            raise ValueError("Control must be +1 or -1")
        
        next_state = max(1, min(10, next_state))  # Ensure the state is within bounds
        
        # Cost function g(x)
        g_x = cost[next_state - 1]
        
        total_cost += (alpha ** k) * g_x
        states.append(next_state)
        current_state = next_state
    
    return states, total_cost


def q_learning(cost, num_episodes=1000000, alpha=0.1, gamma=0.9, epsilon=0.05, max_iter_episode=20):
    """
    Perform Q-learning on the stochastic system.
    
    Parameters:
    - num_episodes: Number of episodes to run.
    - alpha: Learning rate.
    - gamma: Discount factor.
    - epsilon: Exploration rate.
    
    Returns:
    - Q: Learned Q-values.
    """
    Q = np.zeros((10, 2))  # 10 states and 2 actions (+1, -1)
    
    for _ in range(num_episodes):
        state = random.randint(0, 9)  # Start from a random state
        for _ in range(max_iter_episode):
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1])  # Explore
            else:
                action = np.argmin(Q[state])  # Exploit
            
            next_state = state + 1 if action == 0 and random.uniform(0, 1) < 0.5 else state
            next_state = state - 1 if action == 1 and random.uniform(0, 1) < 0.5 else state
            next_state = max(0, min(9, next_state))
            
            reward = cost[next_state]
            
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_delta = td_target - Q[state, action]
            Q[state, action] += alpha * td_delta
            
            state = next_state
    
    return Q



# Example usage
initial_state = 1
cost = [1, 2, 3, 4, 5, 4, 2, 0, 1, 2]
controller = [1, 1, 1, 1, -1, 1, 1, 1, -1, -1]
trajectory, total_cost = simulate_trajectory(initial_state, cost, controller)

print("Trajectory:", trajectory)
print("Total discounted cost:", total_cost)


# Example usage
Q_values = q_learning(cost)
print("Learned Q-values:")
print(Q_values)
