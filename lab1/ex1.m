clc; clear; close all;

%% Initialize Parameters
m = 100;  % mass
a1 = 1;   % friction parameter
a2 = 0.05; % friction parameter
a3 = 0.1; % friction parameter
b = 200;  % motor term

% Define additional parameters as required

%% Design Fuzzy Controller
% Create a new fuzzy inference system
fis = mamfis('Name','TrainController');

% Add input/output variables and their membership functions
fis = addInput(fis,[-1000 1000],'Name','SpeedError');

% Define membership functions for SpeedError
fis = addMF(fis, 'SpeedError', 'trapmf', [-1000 -1000 -1 0], 'Name', 'Negative');
fis = addMF(fis, 'SpeedError', 'trapmf', [-1 0 0 1], 'Name', 'Zero');
fis = addMF(fis, 'SpeedError', 'trapmf', [0 1 1000 1000], 'Name', 'Positive');


%figure()
% Plot Output MFs
%plotmf(fis,'input',1,1000);
%set(findall(gca, 'Type', 'Line'),'LineWidth',3);


% Add output variable
fis = addOutput(fis,[-1 1],'Name','MotorForce');

% Define membership functions for MotorForce (example)
fis = addMF(fis, 'MotorForce', 'trapmf', [-1 -1 -0.5 -0.2], 'Name', 'Low');
fis = addMF(fis, 'MotorForce', 'trapmf', [-0.2 -0.1 0.1 0.2], 'Name', 'Medium');
fis = addMF(fis, 'MotorForce', 'trapmf', [0.2 0.5 1 1], 'Name', 'High');

% Define Rules
rule1 = "If SpeedError is Negative then MotorForce is Low";
rule2 = "If SpeedError is Zero then MotorForce is Medium";
rule3 = "If SpeedError is Positive then MotorForce is High";

fis = addRule(fis, rule1);
fis = addRule(fis, rule2);
fis = addRule(fis, rule3);

%figure()
% Plot Output MFs
%plotmf(fis,'output',1,1000);
%set(findall(gca, 'Type', 'Line'),'LineWidth',3);

%fuzzyLogicDesigner(fis);

%% System Simulation
% Define the time span and initial conditions
dt = 0.01;
timeSpan = 0:dt:500; % Time vector
initialConditions = [0; 0]; % Initial conditions

% Define the ODE function using fuzzy controller
odeFunc = @(t, x) trainSystemDynamics(t, x, fis);

% Run the simulation
[t, x] = ode45(odeFunc, timeSpan, initialConditions);
acc = diff(x(:, 2)) / dt;

motforce = evalfis(fis, referenceSpeed(x(:,1))-x(:,2));

figure;
plot(t, motforce);

% Plot results
figure;
plot(t, x(:,1)); % Plot position
hold on;
plot(t, x(:,2)); % Plot speed
hold on;
plot(t(1:end-1) + dt/2, acc/9.8); % Plot acceleration
xlabel('Time (s)');
ylabel('Position, Speed and Acceleration');
legend({'Position', 'Speed', 'Acceleration'});
title('Train System Response with Acceleration');

%% Train System Dynamics Function
function dx = trainSystemDynamics(t, x, fis)
    global b a1 a2 a3 m

    % Initialize dx
    dx = zeros(2,1);

    % Extract states
    position = x(1);
    speed = x(2);

    % Controller Input
    % Speed error calculation
    speedError = referenceSpeed(position) - speed;

    % Fuzzy Controller Output
    motorForce = evalfis(fis, speedError);

    % System Dynamics
    dx(1,1) = speed; % dx/dt = speed 
    dx(2,1) = (b * motorForce - a1 * speed - a2 * m * speed - a3 * speed * abs(speed)) / m; % Update acc
end

%% Reference Speed Function
function refSpeed = referenceSpeed(p)
    % Define the reference speed function as per exercise requirements
    % Example:
    refSpeed = 10*trapmf(p,[-0.1 100 4900 5000]); % Constant reference speed
end
