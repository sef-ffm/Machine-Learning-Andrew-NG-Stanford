function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Zero-matrix with k columns
X_temp = zeros(size(X, 1),3);   

% Find euclidian distance to each centroid
for i = 1:K;
    X_temp(:, i) = sqrt(sum(power(X - centroids(i, :), 2), 2)); 
end;

% Find location of the minimum value and copy it to idx(i)
[vMax, locMax] = min(X_temp');
idx = locMax';

%
% =============================================================

end

