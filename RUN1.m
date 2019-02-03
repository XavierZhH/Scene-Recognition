clc;

clear;

 

% storage paths of training images and test images

trainDir = 'C:\Users\hz2m18\Downloads\training';

testDir = 'C:\Users\hz2m18\Downloads\testing';

 

% If this value is true, the processed training images and labels information are read from the file; 

% if this value is false, the training images will be preprocessed from the beginning and saved as a .mat file.

loadTrainData = false;

processTrainData(trainDir, 16, loadTrainData);    

trainData = load('run1_trainImgs_trainLabels.mat'); % load the file to get the preprocessed images information matrix and label matrix

trainImgs = trainData.trainImgs;

trainLabels = trainData.trainLabels;

labelNs = load('run1_labelNames.mat'); % load the file to get the correspondence between category label numbers and category strings

labelNames = labelNs.labelNames;

 

% Same as above, preprocessing test data or reading directly from the file

loadTestData = false;

processTestData(testDir, 16, loadTestData);

testData = load('run1_testImgs_testNames.mat');

testImgs = testData.testImgs;

testNames = testData.testNames;

 

[trainRow, ~] = size(trainImgs);% row vector

[testRow, ~] = size(testImgs);

k = floor(sqrt(trainRow));  % select the k value, here is the square root of the training data set size

disp(k);

fp = fopen('run1.txt', 'w');%"write"

for i=1:testRow % predict using KNN and write results to txt file

    predictLabel = KNN(testImgs(i,:), trainImgs, trainLabels, k);   % predict the category of a test image using the KNN algorithm

    labelName = strtrim(labelNames(predictLabel, :)); % get the category name corresponding to the category number

    imgName = strtrim(testNames(i,:));  % get the file name of the test image

    fprintf(fp, '%s %s \r\n', imgName, labelName); % write filename-category name to the result txt file.

    if mod(i, 200) == 0 % print progress every 200 test imgs

        fprintf('Done %d/%d\n', i, testRow);

    end

end

fclose(fp);

 


 



 

