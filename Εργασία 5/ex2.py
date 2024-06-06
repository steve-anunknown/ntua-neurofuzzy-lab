import numpy as np
import matplotlib.pyplot as plt

# Number of random points to generate
num_points = 10000000

# Generate random points
x = np.random.uniform(low=0, high=1, size=num_points)
y = np.random.uniform(low=0, high=1, size=num_points)

# Calculate the number of points within the unit circle
inside_circle = np.sum(x**2 + y**2 <= 1)

# Estimate of pi
pi_estimate = 4 * inside_circle / num_points
print('pi ~', pi_estimate)


# Plot the first 1000 points with the unit circle
plt.figure()
plt.scatter(x[:1000], y[:1000], s=1)
theta = np.linspace(0, np.pi/2, 100)
plt.plot(np.cos(theta), np.sin(theta), 'r')
# set limits to the plot
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('First 1000 points with the unit circle')
plt.show()