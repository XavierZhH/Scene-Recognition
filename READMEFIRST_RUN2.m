% Instrcution: 
%       There are nine parts in our algorithm of 'K-Means', they are 
%           natsort, natsortfiles, sortObj, sigmoid, lrCostFunction, 
%           fmincg, oneVsAll, predictOneVsAll and run2. 'run2' includes the
%           functions of processTrainData, processTestData, visualFeatures.


%       'natsort.m', 'natsortfiles.m' and 'sortObj.m' are taken from 
%           http://cn.mathworks.com/matlabcentral/fileexchange/34464-customizable-natural-order-sort
%           by Stephen Cobeldick, and it is used to let images sort by 
%           natural-order, like 0 1 2 3 4 ... They are called in
%           processTestData.

%       'sigmoid.m' is used to be the activation function, and this is called
%           in 'lrCostFunction.m'.

%       'lrCostFunction.m' is to minimize continuous differentialble multivariate 
%           function, and it is used to calculate the loss function and the 
%           gradient with a regular term, and regular terms generally do 
%           not punish bias terms. This is called in 'oneVsAll.m'.

%       'fmincg.m' is used to used to get the optimal direction of
%           weight, and this is called in 'oneVsAll.m'.

%       'oneVsAll.m' is used to train 15 logistic regression classifiers,
%           and calculates the final theta. This is called in 'run2.m'.

%       'predictOneVsAll.m' is used to predict which class the input is,
%           and this is called in 'run2.m'

%       'run2.m' is the main program, it combines all the above parts, and 
%           also includes the kmeans algorithm, reading the images, extract
%           the features of every image using small patches, converting 
%           images to bag-of-words vectors and storing the result as a txt
%           file.


% Usage:
%       'sigmoid.m':
%           g = sigmoid(z)
%               Parameters:
%                   'z' is the input, and this function computes the sigmoid 
%                       of z.

%       'lrCostFunction.m':
%           [J, grad] = lrCostFunction(theta, X, y, lambda)
%               Parameters:
%                   'theta' is the input direction of weigth,
%                   'X' is the input deasign matrix,
%                   'y' is the true label of corresponding X,
%                   'lambda' is the regularization parameter.

%       'fmincg.m':
%           [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
%               Parameters: 
%                   'f' is function for calculating the function value and 
%                       the partial derivatives,
%                   'X' is the starting point, an array of the weights. 
%                       The best direction of weight will start looking for 
%                       from this vector,
%                   'options' is the maximum number of line searches if it 
%                       is positive, or maximum allowed number of function 
%                       evaluations if negative.

%       'oneVsAll.m':
%           [all_theta] = oneVsAll(X, y, num_labels, lambda, maxIters)
%               Parameters: 
%                   'X' is the input matrix with features,
%                   'y' is the true label of corresponding input features,
%                   'num_labels' is the number of categories,
%                   'lambda' is the regularization parameter,
%                   'maxIters' gives the maximum number of line searches.

%       'predictOneVsAll.m':
%           p = predictOneVsAll(all_theta, X)
%               Parameters: 
%                   'all_theta' is the final weight we got in 'oneVsAll.m',
%                   'X' is the input matrix with features.

%       Function of 'processTrainData' in 'run2.m':
%           processTrainData(trainDir, targetSize, patchSize, stepSize, load)
%               Parameters:
%                   'trainDir' is the path of the training images,
%                   'targetSize' is the size reduce the picture from around
%                       to,
%                   'patchSize' and 'stepSize' are the scope of the
%                       features extracted and the movement between every
%                       feature,
%                   'load' is true means the processed training images and 
%                       labels information are read from the file; 
%                   'load' is false means the training images will be 
%                       preprocessed from the beginning and saved as 
%                       a .mat file. 

%       Function of 'processTestData' in 'run2.m':
%           processTestData(C, k, testDir, targetSize, patchSize, stepSize, load)
%               Parameters:
%                   'C' is the cluster centroid locations in Kmeans algorithm,
%                   'k' is the number of clusters in Kmeans algorithm
%                   'testDir' is the path of the testing images, 
%                   'targetSize' is the size reduce the picture from around
%                       to,
%                   'patchSize' and 'stepSize' are the scope of the
%                       features extracted and the movement between every
%                       feature,
%                   'load' is true means the processed testing images and 
%                       labels information are read from the file; 
%                   'load' is false means the testing images will be 
%                       preprocessed from the beginning and saved as 
%                       a .mat file. 

%       Function of 'visualFeatures' in 'run2.m':
%           imgFeatures = visualFeatures(img, patchSize, stepSize)
%               Parameters:
%                   'img' is the input images,
%                   'patchSize' and 'stepSize' are the scope of the
%                       features extracted and the movement between every
%                       feature.

%       'run2n.m':
%           All parameters of above functions can be changed in this, and  
%               you can run the program, and then will get a text file 
%               named 'run2.txt', all the results of the classification of 
%               the test images are shown in this text file.

% Written by Hang Zhong and Haojiong Wang.