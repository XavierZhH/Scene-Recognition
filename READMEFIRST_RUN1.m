% Instrcution: 
%       There are eight parts in our algorithm of 'KNN', they are processTrainData, 
%           natsort, natsortfiles, sortObj, processTestData, 
%           cosineDistance, KNN and RUN1.

%       'processTrainData.m' is used to process training data, including resize,
%           converting to vectors, zero mean and uint length, etc. 

%       'natsort.m', 'natsortfiles.m' and 'sortObj.m' are taken from 
%           http://cn.mathworks.com/matlabcentral/fileexchange/34464-customizable-natural-order-sort
%           by Stephen Cobeldick, and it is used to let images sort by 
%           natural-order, like 0 1 2 3 4 ... They are called in
%           processTestData.

%       'processTestData.m' is used to process testing data for
%       predictions.

%       'cosineDistance.m' is used to calculate cosine similarity.

%       'KNN.m' is used to predict testing images using KNN algorithm.

%       'RUN1.m' is the main program, it combines all the above parts, and 
%           all parameters can be changed in 'RUN1.m'.


% Usage:
%       'processTrainData.m':
%           processTrainData(trainDir, targetSize, load)
%               Parameters:
%                   'trainDir' is the path of the training images,
%                   'targetSize' is the size reduce the picture from around
%                       to,
%                   'load' is true means the processed training images and 
%                       labels information are read from the file; 
%                   'load' is false means the training images will be 
%                       preprocessed from the beginning and saved as 
%                       a .mat file. 

%       'processTestData.m':
%           processTestData(testDir, targetSize, load)
%               Parameters:
%                   'testDir' is the path of the testing images, 
%                   'targetSize' is the size reduce the picture from around
%                       to,
%                   'load' is true means the processed testing images and 
%                       labels information are read from the file; 
%                   'load' is false means the testing images will be 
%                       preprocessed from the beginning and saved as 
%                       a .mat file. 

%       'cosineDistance.m':
%           cosineSimilarity = cosineDistance(matA,matB)
%               Parameters: 
%                   'matA' and 'matB' are both the input matrices

%       'KNN.m':
%           resultLabel = KNN(input,data,labels,k)
%               Parameters: 
%                   'input' is what we want to test, 
%                   'data' is the training matrices, 
%                   'labels' is the catagories of the training matrices,
%                   'k' is the number of nearest neighbors in the
%                       predictors.

%       'RUN1.m':
%           All parameters of above functions can be changed in this, and  
%               you can run the program, and then will get a text file 
%               named run1.txt, all the results of the classification of 
%               the test images are shown in this text file.

% Written by Hang Zhong and Haojiong Wang.