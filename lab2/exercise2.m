% Generate the data
n = 100; % number of measurements
p = 80; % dimension of x
x = randn(n, p); % xs from a standard normal distribution
c = [rand(10, 1); zeros(60, 1); rand(10, 1)]; % c with 20 non-zero elements
e = randn(n, 1) * 0.5; % e_i uniformly distributed normal random variables
y = x * c + e; % y_i

% Ordinary Least Squares Estimation
c_ols = (x' * x) \ (x' * y);

% Display OLS estimate
disp('OLS Estimate of c:');
disp(c_ols');

% Lasso Regression with Cross-Validation
[c_lasso, FitInfo] = lasso(x, y, 'CV', 10); % 10-fold cross-validation

% Choosing the best lambda
lambda_optimal = FitInfo.Lambda1SE;
c_lasso_optimal = c_lasso(:, FitInfo.Index1SE);
disp('Optimal Lambda:');
disp(lambda_optimal);
disp('Lasso Estimate with Optimal Lambda:');
disp(c_lasso_optimal);

