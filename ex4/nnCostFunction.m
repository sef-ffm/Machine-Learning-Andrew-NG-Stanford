function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1 Feedforward the neural network and return the cost in the variable J.

% Forward propagation
a_1 = [ones(m,1) X];                % first layer, add ones 5000 x 401
a_2 = sigmoid(Theta1 * a_1');       % second layer (25x401) * (5000x401)' = 25x5000
a_2 = cat(1, ones(1, m), a_2);      % second layer, add ones 26x5000
a_3 = sigmoid(Theta2 * a_2 );       % third layer 10x26 * 26x5000 = 10x5000

% Recode the (original) labels as vectors containing only values 0 or 1  
y_labels = zeros(num_labels, m);    % 10*5000
for i=1:m
  y_labels(y(i),i)=1;
end

% Compute the cost 
J = (1/m)*sum(sum((-y_labels.*log(a_3)) - ((1-y_labels).*log(1-a_3))));

% Regulatization term
Theta1_temp = Theta1(:, 2:end);    % We should not regularize the bias (the first column of each matrix)
Theta2_temp = Theta2(:, 2:end);    % We should not regularize the bias (the first column of each matrix)
reg_term = (lambda/(2*m))*(sum(sum(Theta1_temp.^2, 2)) + sum(sum(Theta2_temp.^2,2)));
J = J + reg_term;
   

% Part 2: Implement the backpropagation algorithm to compute the gradients

for t = 1:m
    % Step 1, perform feedforward
    a_1 = X(t, :);                  % set the input layer value to t-th training example
    a_1 = [1, a_1];                 % add one for bias units
    a_2 = sigmoid(Theta1 * a_1');   % perform feedforward pass
    a_2 = [1; a_2];                 % add one for bias units
    a_3 = sigmoid(Theta2 * a_2 );   % perform feedforward pass

    % Step 2, delta 3rd layer
    d_3 = a_3 - y_labels(:, t);          
    
    % Step 3, delta 2rd layer, need to add bias 1
    d_2 = (Theta2' * d_3) .* [1; sigmoidGradient(Theta1 * a_1')]; % (26x1) .* (25x1)
    
    % Step 4, accumulate the gradient
    d_2 = d_2(2:end);                           % skip the first term
    Theta1_grad = Theta1_grad + d_2 * a_1;      % (25x1) * (1x401)
    Theta2_grad = Theta2_grad + d_3 * a_2';     % (10x1) * (1x26)

end
    
    % Step 5, obtain the unregularized gradient
    Theta1_grad = 1/m * Theta1_grad;
    Theta2_grad = 1/m * Theta2_grad;
    
    % Regularization    
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m)*Theta1(:, 2:end);
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m)*Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
