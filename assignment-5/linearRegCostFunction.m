function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

h= X*theta;
c= h-y;
c=c.^2;
c=sum(c);
a=1/(2*m);
c=c*a;
d= lambda/(2*m);
thetas=theta(2:length(theta));
e= sum((thetas.^2));
e=e*d;
J=c+e;



grad = zeros(size(theta));
c= X'*(h-y);
c=c/m;
t= c(2:length(c));
thetas= thetas*(lambda/m);
t=t+thetas;
g= c(1:1);
grad=[g;t];


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
