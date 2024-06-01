% Load data
run('el_load.m')
run('deseasonalization.m')
y = el_lo_des';

% Preprocess data (if necessary)
% e.g., handling missing values, normalization

% Determine the order of the AR model using cross-validation
maxorder = 20;
cvError = zeros(1, maxorder);
for p = 1:maxorder % Test AR orders from 1 to maxorder
    % Split data into training and validation sets
    n = length(y);
    trainData = y(1:round(0.7*n)); % 70% for training
    valData = y(round(0.7*n)+1:end); % 30% for validation

    % Fit AR model of order p
    model = ar(trainData, p);
    
    % Predict on validation data
    predicted = forecast(model, valData, length(valData));
    
    % Calculate error (e.g., MSE)
    cvError(p) = immse(valData, predicted);
end

% Choose the order with the minimum cross-validation error
[~, optimalP] = min(cvError);

% Estimate AR model parameters using the entire dataset
finalModel = ar(y, optimalP);

% Display the model
disp(finalModel)

% Validate the model (check residuals, performance metrics, etc.)
