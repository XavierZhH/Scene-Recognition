%Calculation of loss function and gradient with regular terms

%Regular terms generally do not punish bias terms

function [J, grad] = lrCostFunction(theta, X, y, lambda)

 

m = length(y); % number of training examples,15

 

%Do not punish the first item

J = (1/m)*sum((-y.*log(sigmoid(X*theta)))-(1-y).*log(1-sigmoid(X*theta)))+(lambda/(m*2))*sum(theta(2:end).*theta(2:end));

grad = (1/m)*(X'*(sigmoid(X*theta)-y))+(lambda/m)*theta;

temp=(1/m)*(X'*(sigmoid(X*theta)-y));

grad(1)=temp(1);

 

 

end


