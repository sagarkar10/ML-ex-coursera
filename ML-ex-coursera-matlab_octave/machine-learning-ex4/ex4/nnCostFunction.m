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



% Turn y from a column vector like this [1; 1; 3; 4]
% to  [1 0 0 0 0 0 0 0 0 0]
%     [1 0 0 0 0 0 0 0 0 0]
%     [0 0 1 0 0 0 0 0 0 0]
%     [0 0 0 1 0 0 0 0 0 0]
% Y is a 5000 x 10 matrix.
Y = [];
for i = 1:length(y)
    temp = zeros(1, num_labels);       % create a row vector of zeros
    temp(y(i)) = 1;   % turn on the flag for the value
    if length(Y) == 0
        Y = temp;
    else
        Y = [Y; temp];
    end
end

% Use forward propagation to compute the values of a3
X = [ones(size(X, 1), 1) X]; % 5000 x 401
a2 = sigmoid(X * Theta1');
a2 = [ones(size(a2, 1), 1) a2];
a3 = sigmoid(a2 * Theta2'); % 5000 x 10

% Compute the cost function and the regularization parameter
for i = 1:m
    for k = 1:num_labels
        J =J + -Y(i, k) * log(a3(i, k)) - (1 - Y(i, k)) * log(1 - a3(i, k));
    end
end

%J = (1/m)*sum(sum(-Y.*log(a3) - (1-Y).*(log(1-a3)))); ??? Y NOT THIS?


J = (1 / m) * J;

% Compute the regularization parameter (excluding the bias unit)
J = J + (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

% Compute the gradients using backpropagation
for t = 1:m
    a1 = X(t, :); % 1 x 401
    z2 = a1 * Theta1'; % 1 x 25
    a2 = sigmoid(z2); % 1 x 25
    a2 = [1 a2];     % 1 x 26  - add the bias unit
    z3 = a2 * Theta2'; % 1 x 10
    a3 = sigmoid(z3); % 1 x 10

    d3 = a3 - Y(t, :); % 1 x 10
    d2 = d3 * Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]); % (1 x 10 * 10 x 26) .* 1 x 26

    Theta1_grad = Theta1_grad + (d2(2: end))' * a1; % 25 x 401
    Theta2_grad = Theta2_grad + d3' * a2; % 10 x 26    10 x 26 + (1 x 10 * 26 x 1)
    
end


Theta2_grad = 1 / m * Theta2_grad; % 10 x 26
Theta1_grad = 1 / m * Theta1_grad; % 25 x 401

% Regularize the gradient
reg = lambda / m * Theta2; 
reg(:, 1) = zeros(size(reg, 1), 1); % don't regularize the the first column of theta
Theta2_grad = Theta2_grad + reg;

reg = lambda / m * Theta1;
reg(:, 1) = zeros(size(reg, 1), 1);
Theta1_grad = Theta1_grad + reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end