import numpy as np
import matplotlib.pyplot as plt

# System dynamics of the pendulum
def pendulum_dynamics(x, a1, a2, b, u):
    x1, x2 = x
    x1_dot = x2
    x2_dot = -a1 * np.sin(x1) - a2 * x2 + b * u
    return np.array([x1_dot, x2_dot])

# Gradient descent update
def update_parameters(x, x_hat, a1, a2, b, u, learning_rate):
    x1, x2 = x
    x1_hat, x2_hat = x_hat

    # Compute the gradients
    e1 = x1 - x1_hat
    e2 = x2 - x2_hat

    # Update rules
    a1_grad = -e2 * np.sin(x1_hat)
    a2_grad = -e2 * x2_hat
    b_grad = e2 * u

    a1 += learning_rate * a1_grad
    a2 += learning_rate * a2_grad
    b += learning_rate * b_grad

    return a1, a2, b

# Simulation parameters
dt = 0.01  # time step
T = 10     # total time of simulation
t = np.arange(0, T, dt)

# Initial conditions and true parameters
x0 = [np.pi/4, 0]  # initial state
true_a1, true_a2, true_b = 1.0, 0.5, 0.3  # true parameters of the system

# Initial guesses for the parameters
a1, a2, b = 0.5, 0.25, 0.15  # initial guesses

# Learning rate for the gradient descent
learning_rate = 0.01

# Storage for data
x_storage = np.zeros((len(t), 2))
x_hat_storage = np.zeros((len(t), 2))
a1_storage = np.zeros(len(t))
a2_storage = np.zeros(len(t))
b_storage = np.zeros(len(t))

# Initial state
x = np.array(x0)
x_hat = np.array(x0)  # initial estimate

# Simulation loop
for i in range(len(t)):
    # Assuming a constant input u
    u = 1.0

    # System dynamics
    x_dot = pendulum_dynamics(x, true_a1, true_a2, true_b, u)
    x = x + x_dot * dt

    # Estimated dynamics
    x_hat_dot = pendulum_dynamics(x_hat, a1, a2, b, u)
    x_hat = x_hat + x_hat_dot * dt

    # Parameter update
    a1, a2, b = update_parameters(x, x_hat, a1, a2, b, u, learning_rate)

    # Store data
    x_storage[i] = x
    x_hat_storage[i] = x_hat
    a1_storage[i] = a1
    a2_storage[i] = a2
    b_storage[i] = b

# Plotting the results
plt.figure(figsize=(12, 8))
# title for the whole figure
plt.suptitle('Pendulum System Identification with Gradient Descent (Constant Input U = 1.0)', fontsize=16)
plt.subplot(2, 2, 1)
plt.plot(t, x_storage[:, 0], label='True x1')
plt.plot(t, x_hat_storage[:, 0], label='Estimated x1')
plt.title('State x1')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, x_storage[:, 1], label='True x2')
plt.plot(t, x_hat_storage[:, 1], label='Estimated x2')
plt.title('State x2')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, a1_storage, label='a1')
plt.axhline(y=true_a1, color='r', linestyle='--')
plt.title('Parameter a1')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, a2_storage, label='a2')
plt.axhline(y=true_a2, color='r', linestyle='--')
plt.title('Parameter a2')
plt.legend()

plt.tight_layout()
plt.savefig('estimation-const-unit-input.png')
plt.close()

# ==================================================== #
from scipy.signal import square

# Function to generate a square wave
def square_wave(t, amplitude, frequency):
    return amplitude * square(2 * np.pi * frequency * t)

# Simulation parameters for square wave input
sw_frequency = 0.5  # frequency of the square wave
sw_amplitude = 1.0  # amplitude of the square wave

# Reset initial guesses for the parameters
a1, a2, b = 0.5, 0.25, 0.15  # reset to initial guesses

# Reset initial state
x = np.array(x0)
x_hat = np.array(x0)  # reset initial estimate

# Reset storage for data
x_storage = np.zeros((len(t), 2))
x_hat_storage = np.zeros((len(t), 2))
a1_storage = np.zeros(len(t))
a2_storage = np.zeros(len(t))
b_storage = np.zeros(len(t))

# Simulation loop with square wave input
for i in range(len(t)):
    # Square wave input
    u = square_wave(t[i], sw_amplitude, sw_frequency)

    # System dynamics
    x_dot = pendulum_dynamics(x, true_a1, true_a2, true_b, u)
    x = x + x_dot * dt

    # Estimated dynamics
    x_hat_dot = pendulum_dynamics(x_hat, a1, a2, b, u)
    x_hat = x_hat + x_hat_dot * dt

    # Parameter update
    a1, a2, b = update_parameters(x, x_hat, a1, a2, b, u, learning_rate)

    # Store data
    x_storage[i] = x
    x_hat_storage[i] = x_hat
    a1_storage[i] = a1
    a2_storage[i] = a2
    b_storage[i] = b

# Plotting the results with square wave input
plt.figure(figsize=(12, 8))
plt.suptitle('Pendulum System Identification with Gradient Descent (U = unit square wave)', fontsize=16)
plt.subplot(2, 2, 1)
plt.plot(t, x_storage[:, 0], label='True x1')
plt.plot(t, x_hat_storage[:, 0], label='Estimated x1')
plt.title('State x1 with Square Wave Input')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, x_storage[:, 1], label='True x2')
plt.plot(t, x_hat_storage[:, 1], label='Estimated x2')
plt.title('State x2 with Square Wave Input')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, a1_storage, label='a1')
plt.axhline(y=true_a1, color='r', linestyle='--')
plt.title('Parameter a1 with Square Wave Input')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, a2_storage, label='a2')
plt.axhline(y=true_a2, color='r', linestyle='--')
plt.title('Parameter a2 with Square Wave Input')
plt.legend()

plt.tight_layout()
plt.savefig('estimation-square-wave-input.png')
plt.close()

# ==================================================== #
# Function to generate a sinusoidal wave
def sinusoidal_wave(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

# Simulation parameters for sinusoidal wave input
sin_frequency = 0.5  # frequency of the sinusoidal wave
sin_amplitude = 1.0  # amplitude of the sinusoidal wave

# Reset initial guesses for the parameters
a1, a2, b = 0.5, 0.25, 0.15  # reset to initial guesses

# Reset initial state
x = np.array(x0)
x_hat = np.array(x0)  # reset initial estimate

# Reset storage for data
x_storage = np.zeros((len(t), 2))
x_hat_storage = np.zeros((len(t), 2))
a1_storage = np.zeros(len(t))
a2_storage = np.zeros(len(t))
b_storage = np.zeros(len(t))

# Simulation loop with sinusoidal wave input
for i in range(len(t)):
    # Sinusoidal wave input
    u = sinusoidal_wave(t[i], sin_amplitude, sin_frequency)

    # System dynamics
    x_dot = pendulum_dynamics(x, true_a1, true_a2, true_b, u)
    x = x + x_dot * dt

    # Estimated dynamics
    x_hat_dot = pendulum_dynamics(x_hat, a1, a2, b, u)
    x_hat = x_hat + x_hat_dot * dt

    # Parameter update
    a1, a2, b = update_parameters(x, x_hat, a1, a2, b, u, learning_rate)

    # Store data
    x_storage[i] = x
    x_hat_storage[i] = x_hat
    a1_storage[i] = a1
    a2_storage[i] = a2
    b_storage[i] = b

# Plotting the results with sinusoidal wave input
plt.figure(figsize=(12, 8))
plt.suptitle('Pendulum System Identification with Gradient Descent (U = sin(pi * t))', fontsize=16)
plt.subplot(2, 2, 1)
plt.plot(t, x_storage[:, 0], label='True x1')
plt.plot(t, x_hat_storage[:, 0], label='Estimated x1')
plt.title('State x1 with Sinusoidal Input')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, x_storage[:, 1], label='True x2')
plt.plot(t, x_hat_storage[:, 1], label='Estimated x2')
plt.title('State x2 with Sinusoidal Input')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, a1_storage, label='a1')
plt.axhline(y=true_a1, color='r', linestyle='--')
plt.title('Parameter a1 with Sinusoidal Input')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, a2_storage, label='a2')
plt.axhline(y=true_a2, color='r', linestyle='--')
plt.title('Parameter a2 with Sinusoidal Input')
plt.legend()

plt.tight_layout()
plt.savefig('estimation-sin-input.png')
plt.close()



