function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); %store the values of each iteration in a vector

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %J = cosmputeCost(X,y,theta);
    theta = theta - ((alpha/m)*(X' * (X*theta - y)));
    % we directly vectorised it so we did it for multiple as well as this
    % case, X is a mx2 matrix and the (X*theta -y) is a m*1 col vec so we
    % multiplied it thats way
    %figure;
    %hold on; % keep previous plot visible  these already done
    
    %plot(X(:,2), X*theta, '-')  %plotting 2nd (non bias) feature with the predictions made by the selected theta 
    %legend('Training data', 'Linear regression')
    %hold off;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
