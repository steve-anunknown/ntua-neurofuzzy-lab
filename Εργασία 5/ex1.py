import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Vehicle parameters and initial state (assuming some values)
V = 1 # Velocity of the vehicle in m/s
T_s = 0.1 # Sampling time in seconds
y_0 = 1 # Initial lateral displacement in meters
Vy_0 = 0 # Initial lateral velocity in m/s

# System matrices (assuming a simple model)
A = np.array([[1, 0],
              [V*T_s, 1]])
B = np.array([[T_s],
              [0.5*T_s**2]])

# Control horizon
N = 10 # Number of time steps for the control horizon

# Define the decision variables
u = cp.Variable(N+1) # Control input (lateral acceleration)

# States over the horizon
y = cp.Variable(N+1)
Vy = cp.Variable(N+1)

# Maximum allowable lateral acceleration
U_max = 1 # Assuming a maximum of 0.3 m/s^2 for lateral acceleration


# Weights for the cost function
lambdas_1 =[0.1, 1, 10]
lambdas_2 = [0.1, 1, 10]

for lambda_1, lambda_2 in product(lambdas_1, lambdas_2):

    # Cost function
    t = 1
    cost = lambda_1*cp.abs(Vy[t+3]) + lambda_2*cp.abs(y[t+3])
    for k in range(N):
        cost += lambda_1*cp.abs(Vy[t+k]) + lambda_2*cp.abs(y[t+k]) + cp.abs(u[t+k])

    # Constraints
    constraints = [y[0] == y_0, Vy[0] == Vy_0]
    for k in range(N):
        constraints.append(y[k+1] == A[0][0]*y[k] + A[0][1]*Vy[k] + B[0]*u[k])
        constraints.append(Vy[k+1] == A[1][0]*y[k] + A[1][1]*Vy[k] + B[1]*u[k])
        constraints.append(cp.abs(u[k]) <= U_max)

    # Setup the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # Solve the optimization problem
    problem.solve(solver=cp.ECOS_BB)

    # Extract the optimal control sequence
    u_optimal = u.value
    y_optimal = y.value
    Vy_optimal = Vy.value

    fig, axs = plt.subplots(3)

    # Set title
    fig.suptitle(f'λ\u2081: {lambda_1}, λ\u2082: {lambda_2}')

    # Plot the optimal control sequence
    axs[0].plot(u_optimal)
    axs[0].set(xlabel='Time step', ylabel='Lateral acceleration (m/s^2)', title='Optimal control sequence')

    # Plot the optimal state sequence
    axs[1].plot(y_optimal)
    axs[1].set(xlabel='Time step', ylabel='Lateral displacement (m)', title='Optimal state sequence')

    # Plot the optimal state sequence
    axs[2].plot(Vy_optimal)
    axs[2].set(xlabel='Time step', ylabel='Lateral velocity (m/s)', title='Optimal state sequence')

plt.tight_layout()
plt.show()
