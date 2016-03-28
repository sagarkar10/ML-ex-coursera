function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% my note
% theta is a col vec 2 x 1
% X is m x 2 vector, so X*theta is cool to produce m x 1 prediction 
% j = 1/2m.sum(ho(x)-y)^2

J = sum((1/2*m)*((X*theta - y).^2)); % square each term of the matrix

% =========================================================================

end