data = readtable('Dataset 1.csv');
disp(data.Properties.VariableNames);

% 2. Feature Selection
X = data{:, 1:end-1}; % Features
y = data{:, end}; % Target variable

% 3. Splitting the Data
X_train = X(1:end/2, :);
y_train = y(1:end/2);
X_test = X(end/2+1:end, :);
y_test = y(end/2+1:end);

% Set the random seed for reproducibility
rng(42);

% Splitting the Data
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% train, 20% test
idx_train = training(cv);
idx_test = test(cv);

X_train = X(idx_train, :);
y_train = y(idx_train);
X_test = X(idx_test, :);
y_test = y(idx_test);

% 4. Building the Model
model = fitlm(X_train, y_train);

% 6. Model Evaluation
y_pred = predict(model, X_test);
mse = immse(y_test, y_pred);
r_squared = corr(y_test, y_pred)^2;
fprintf('Mean Squared Error: %.4f\n', mse);
fprintf('R-squared: %.4f\n', r_squared);

% Define the metrics and their values
metrics = {'Mean Squared Error', 'R-squared'};
values = [0.5435, 0.9876];

% Plot the bar chart
figure;
bar(values);
xticks(1:numel(metrics));
xticklabels(metrics);
ylabel('Value');
title('Model Evaluation Metrics');

% Add text labels above each bar
text(1:numel(metrics), values, num2str(values', '%.4f'), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

% Show the plot
grid on;


% Perform residual analysis
residuals = y_test - y_pred;

% Plot residuals
figure;
plotResiduals(model);

% Diagnostic plots
figure;
plotSlice(model);


% % Plot the actual data points
% scatter(X_test, y_test, 'b', 'filled'); % Scatter plot of test data
% hold on;
% 
% % Plot the model's prediction curve
% X_range = [min(X_test), max(X_test)]; % Range of feature values
% y_pred_range = predict(model, X_range); % Predicted values for the range
% plot(X_range, y_pred_range, 'r', 'LineWidth', 2); % Plot prediction curve
% 
% % Add labels and legend
% xlabel('Feature');
% ylabel('Target');
% title('Model Prediction Curve');
% legend('Actual Data', 'Prediction Curve');
% 
% % Show the plot
% hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%For Continous new data addition and retraining the Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % 1. Define a function or mechanism to add new data to your dataset
% function addNewData(newData)
%     % Append new data to your existing dataset
%     data = [data; newData];
% end
% 
% % 2. Define a function to retrain the model with updated data
% function retrainModel(data)
%     X = data{:, 1:end-1}; % Features
%     y = data{:, end}; % Target variable
% 
%     % Retrain the model
%     newModel = fitlm(X, y);
% 
%     % Update the global model variable
%     model = newModel;
% end
% 
% % Main loop or event-driven mechanism to continuously check for new data
% while true
%     % Check for new data
%     if newDataAvailable
%         % Add new data to the dataset
%         addNewData(newData);
% 
%         % Retrain the model with updated data
%         retrainModel(data);
% 
%         % Optionally, perform model evaluation or other tasks
%     end
% 
%     % Optionally, add a delay or use event-driven mechanisms
%     pause(60); % Check for new data every 60 seconds
% end
