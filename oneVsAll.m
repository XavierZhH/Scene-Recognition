%Calculate the final theta

%num_labels indicates the number of categories identified

%lambda is a regular item parameter

function [all_theta] = oneVsAll(X, y, num_labels, lambda, maxIters)

 

m = size(X, 1);% return num of row

n = size(X, 2);% return num of col

 

all_theta = zeros(num_labels, n + 1);

 

% Plus 1 column offset, all 1

X = [ones(m, 1) X];

 

%initial_theta, all 0 matrix

initial_theta = zeros(n + 1, 1);

%     

%     % Set options for fminunc

options = optimset('GradObj', 'on', 'MaxIter', maxIters);

 

%For each number to be identified, calculate a column of theta

for c=1:num_labels

    [theta] =  fmincg (@lrCostFunction,initial_theta, options,X, y==c, lambda);

    % The i-th line all_theta'*X is used to identify whether X is belonged to label i

    all_theta(c,:)=theta';

end

