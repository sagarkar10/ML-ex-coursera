function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

hx = sigmoid (X*theta);


% cost1 = -y.*log(hx);        %multiplication must be one to one.... not mat mulp
% cost2 = -(1-y).*log(1-hx); % '' so we need to sum up this
% single_cost = cost1+cost2;
% J = sum(single_cost); 


cost1 = -(y' * log(hx));        %transposing so no sum
cost2 = -((1-y)' * log(1-hx)); % '' so we need not sum up this
J = cost1+cost2; % without regularization

grad = (1/m)*( X' * (hx-y));    %no need to sum as each example is being summed automatically
% without regularization


% sumassion of all
%% =====REGULARIZATION PART=====

% we dont regularize the bias feature
%filter_theta = theta;  %we dont need the bias theta... so we set it to 0 in filter
%filter_theta(1) = 0; 
%costRegularization = (lamda/(2*m))*(sum(theta.*theta));     % sum way
%costRegularization = (lambda/(2*m))*((filter_theta'*filter_theta));     % transpose way
      %sum all becoz there isnt any transpose so
      
%J = cost1+cost2 + costRegularization; % with regularization

% use filter theta as it has first term bias theta zero
% grad = (1/m)*( X' * (hx-y)) + (lambda/m)*filter_theta



% =============================================================

end
