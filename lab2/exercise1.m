% Load data
run('el_load.m')
run('deseasonalization.m')
y = el_lo_des';
n = length(y);
train = round(0.7*n);
rest = round(0.3*n);
% Split data into training and validation sets
trainData = y(1:train); % 70% for training
valData = y(train+1:end); % 30% for validation

% Determine the order of the AR model using cross-validation
maxorder = 20;
cvError = zeros(1, maxorder);
for p = 1:maxorder % Test AR orders from 1 to maxorder
    % Fit AR model of order p
    model = ar(trainData, p);
    % Predict on validation data
    predictions = forecast(model, trainData, rest);
    % Calculate error
    cvError(p) = immse(predictions, valData);
end

% Choose the order with the minimum cross-validation error
[~, optimalP] = min(cvError);

% Estimate AR model parameters using the entire dataset
finalModel = ar(y, optimalP);

% Display the model
disp(finalModel)
disp(optimalP)
disp(cvError)
