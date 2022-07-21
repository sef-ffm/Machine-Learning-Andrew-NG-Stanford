function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% create a table with combinations of parameters and a 'dummy' column for errors 
    C_test      = [0.01 0.03 0.1 0.3 1 3 10 30];
    sigma_test  = [0.01 0.03 0.1 0.3 1 3 10 30];
    table_test  = combvec(C_test, sigma_test)';
    table_test  = [table_test zeros(size(table_test,1),1)];

% loop through combinations, train the model, apply model to 
% validation set, calculate validation errors and store them

for i = 1:length(table_test);
      model = svmTrain(X, y, table_test(i, 1), @(x1, x2) gaussianKernel(x1, x2, table_test(i, 2)));
      predictions = svmPredict(model, Xval);
      table_test(i, 3) = mean(double(predictions ~= yval));
end

% find minimal error and return C and sigma
[minval index_minval] = min(table_test(:,3));
C = table_test(index_minval, 1);
sigma = table_test(index_minval, 2);

% =========================================================================

end
