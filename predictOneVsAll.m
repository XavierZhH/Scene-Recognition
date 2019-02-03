function p = predictOneVsAll(all_theta, X)

 

m = size(X, 1);

num_labels = size(all_theta, 1);

 

p = zeros(size(X, 1), 1);

 

X = [ones(m, 1) X];

 

[a,p]=max(all_theta*X',[],1);%max in col 

 

p=p';

% Hint: This code can be done all vectorized using the max function.

%       In particular, the max function can also return the index of the 

%       max element, for more information see 'help max'. If your examples 

%       are in rows, then, you can use max(A, [], 2) to obtain the max 

%       for each row.

%       

 

 

 

 

 

 

 

% =========================================================================

 

 

end

