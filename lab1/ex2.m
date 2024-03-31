clc; clear; close all;

% Define matrices A1 and A2
syms a;
A1 = [0.9, a; 0, 0.8];
A2 = [0.9, 0; a, 0.8];

% Define the symbolic variable x and matrix P for Lyapunov function
syms x [2 1]
syms P [2 2];

% Assuming P is a positive definite matrix, such as an identity matrix
P = eye(2);

% Lyapunov function V(x) = x'*P*x
V = x'*P*x;

% Derivative of V along the trajectories of the system
V_dot = simplify(gradient(V, x)' * (A1*x + A2*x));

% Solve for conditions on 'a' for stability
stability_conditions = solve(V_dot < 0, a);

% Display results
disp('Stability conditions for a:');
disp(stability_conditions);

% Numerical analysis for specific values of 'a'
% Assuming h1(x) and h2(x) such that h1(x) + h2(x) = 1
h1 = @(x) 0.5; % Example function
h2 = @(x) 0.5; % Example function

% Define a range of 'a' values to test
a_values = -2:0.1:2;

% Initialize array to store stability results
stability_results = zeros(size(a_values));

% Loop over 'a' values
for i = 1:length(a_values)
    a = a_values(i);
    
    % Redefine A1 and A2 with the current 'a' value
    A1 = [0.9, a; 0, 0.8];
    A2 = [0.9, 0; a, 0.8];
    
    % Define the system's dynamics
    systemDynamics = @(t, x) h1(x) * A1 * x + h2(x) * A2 * x;

    % Simulate the system
    [t, x] = ode45(systemDynamics, [0, 10], [0; 0]); % Initial conditions [0; 0]
    
    % Check if the system is stable
    if all(abs(x(end,:)) < 1e-3) % Threshold for stability
        stability_results(i) = 1;
    end
end

% Plot results
figure;
plot(a_values, stability_results, 'b*-');
xlabel('a');
ylabel('Stability (1 for stable, 0 for unstable)');
title('Stability Analysis over Different Values of a');
