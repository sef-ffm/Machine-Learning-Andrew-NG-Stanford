function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% Calculate the cost function
    J = 1/2 * sum(sum(power((X * Theta' - Y).*R, 2)));

% Add regularization parameter to J
    reg_param_a = (lambda/2) * sum(sum(power(Theta, 2)));
    reg_param_b = (lambda/2) * sum(sum(power(X, 2)));
    J = J + reg_param_a + reg_param_b;    


% Gradients of the cost function, Notes:
    % Y = ratings given by 4 users to 5 movies
    % R = ratings by 4 users to 5 movies (rated or not)
    % X = 3 features for 5 movies
    % Theta = thetas for 3 features for 4 users

% Looping over movies to compute X for each movie

    for i = 1:num_movies
    
        % index of users that rated the i-th movie
        idx = find(R(i,:)==1);
        
        % filter thetas of users that rated the i-th movie
        Theta_temp = Theta(idx,:);
        
        % filter ratings of users that rated the i-th movie
        Y_temp = Y(i,idx);
        
        % record the derivative
        X_grad(i, :) = (X(i, :) * Theta_temp' - Y_temp) * Theta_temp;
        
        % Add regularization parameter to gradients
        X_grad(i, :) = X_grad(i, :) + (lambda * X(i, :));
    
    end
    
% Looping over users to compute theta for each user

    for i = 1:num_users
    
        % index of movies rated by i-th user
        idx = find(R(:, i)==1);
        
        % filter features of movies rated by i-th user
        X_temp = X(idx, :);
        
        % filter the thetas for i-th user
        Theta_temp = Theta(i,:);
        
        % filter ratings of movie rated by i-th user
        Y_temp = Y(idx, i);
        
        % record the derivative
        Theta_grad(i, :) = X_temp' * (X_temp * Theta_temp' - Y_temp);
    
        % Add regularization parameter to gradients
        Theta_grad(i, :) = Theta_grad(i, :) + (lambda * Theta(i, :));   
        
    end


    


    
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
