import numpy as np


# dynamics of vehicle
# x_dot = Vcos(a)
# y_dot = Vsin(a)
# a_dot = u
# x is the direction of the lane
# y is the direction perpendicular to the lane
# a is the angle of the vehicle with respect to the x axis
# u is the control input, the steering angle of the vehicle
# V is the velocity

U = 1
Ts = 0.1
V = 1
A = np.array([[1, 0], [V*Ts, 1]])
B = np.array([Ts, 1/2*V*Ts**2])

def car_dynamics(x, u):
    return np.dot(A, x) + B*u
    
def u_constraint(u):
    return u <= U
