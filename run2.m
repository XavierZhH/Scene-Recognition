clc;

clear;

 

trainDir = 'C:\Users\hz2m18\Downloads\training';

testDir = 'C:\Users\hz2m18\Downloads\testing';

 

% Some hyperparameters

% If set too large here, kmeans clustering is very time consuming and consumes a lot of memory.

% Note that if the hyperparameters here need to be modified, change the loadOrNot to false and re-run!

targetSize = 32;    % Image compressed size

patchSize = 8;% The size of each patch when extracting features

stepSize = 4;   % Step size for the patch movement when extracting features

k = 500;    % The k value of the kmeans cluster, which is the size of the visual word vocabulary

 

% Preprocess training images, save them to a file or read directly from a file

loadTrain = false;

processTrainData(trainDir, targetSize, patchSize, stepSize, loadTrain);

loadData = load('run2_trainLabels_labelNames.mat');

trainLabels = loadData.trainLabels;

labelNames = loadData.labelNames;

 

loadDataFeatures = load('run2_allTrainImgFeatures_eachTrainImgFeatures.mat');

allFeatures = loadDataFeatures.allFeatures;

eachImgFeatures = loadDataFeatures.eachImgFeatures;

 

% kmeans clustering

% Perform kmeans clustering on the features of all training images to generate k cluster centroids, that is, visual word vocabulary of size k

loadKMeans = false;

if ~loadKMeans

    % Kmeans up to 1000 iterations, repeated clustering 8 times to reduce the numerical oscillation

    [idx, C, sumd, D] = kmeans(allFeatures, k, 'MaxIter',1000, 'Display','final', 'Options',statset('UseParallel', 1), 'Replicates',8);

else

    kMeansData = load('run2_kmeans.mat');

    idx = kMeansData.idx;

    C = kMeansData.C;

    sumd = kMeansData.sumd;

    D = kMeansData.D;

end

 

[eachImgFeaturesNum, ~, trainImgsNum] = size(eachImgFeatures);

[allFeaturesNum, ~] = size(allFeatures);

 

[~, index] = min(D, [], 2); % Get the cluster of each extracted feature of each training picture

 

% converting features of train images to bag-of-words vectors

trainBoWs = [];

for i=1:trainImgsNum

    bow = zeros(1, k);

    for j=1:eachImgFeaturesNum  % For each image, based on the extracted features and visual word vocabulary, convert them to word bag vectors

        wordIndex = index((i-1)*eachImgFeaturesNum + j);

        bow(1,wordIndex) = bow(1,wordIndex) + 1;

    end

    trainBoWs = [trainBoWs;bow];

end

 

loadTest = false;

processTestData(C, k, testDir, targetSize, patchSize, stepSize, loadTest);  % Convert test data into word bag vectors

testData = load('run2_testBoWs_testNames.mat');

testBoWs = testData.testBoWs;

testNames = testData.testNames;

 

% predict test images and write results to txt file

allTheta = oneVsAll(trainBoWs, trainLabels, 15, 0.1, 200);

p = predictOneVsAll(allTheta, testBoWs);

 

fp = fopen('run2.txt', 'w');

for i=1:size(p, 1)

    label = p(i);

    imgName = strtrim(testNames(i, :));

    labelName = strtrim(labelNames(label, :));

    fprintf(fp, '%s %s \r\n', imgName, labelName);

end

fclose(fp);

 

% converting train images to visual features for kmeans clustering

function processTrainData(trainDir, targetSize, patchSize, stepSize, load)

if ~load

    listDir = dir(trainDir);

    fileNum=size(listDir, 1);

    

    allFeatures = [];

    eachImgFeatures = [];

 

    trainLabels = [];

    labelNames = [];

    for i=1:fileNum

        label = listDir(i).name;

        if ~contains(label, '.')

            disp(label);

            subDir = strcat(trainDir,'\',label);

            subList = dir(fullfile(subDir, '*.jpg'));

            for j=1:length(subList)

                img = im2double(imread(fullfile(subDir, subList(j).name)));% convert image to a pixel matrix

                if size(img, 3)==3

                    img = rgb2gray(img);

                end

                img = imresize(img, [targetSize, targetSize]);  % resize image

 

                imgFeatures = visualFeatures(img, patchSize, stepSize); % extract features of this image

                allFeatures = [allFeatures; imgFeatures];

                eachImgFeatures = cat(3, eachImgFeatures, imgFeatures);

                trainLabels = [trainLabels;i];

            end

        end

        if isempty(labelNames)

            labelNames = label;

        else

            labelNames = char(labelNames, label);

        end

    end

    save run2_trainLabels_labelNames.mat trainLabels labelNames;

    save run2_allTrainImgFeatures_eachTrainImgFeatures.mat allFeatures eachImgFeatures;

else

    disp('Please load .mat file!');

end

end

 

% converting test images to bag-of-words vectors

function processTestData(C, k, testDir, targetSize, patchSize, stepSize, load)

if ~load

    

    testBoWs = [];

    testNames = [];

    

    imgFiles = dir(fullfile(testDir, '*.jpg'));
    imgFiles = sortObj(imgFiles);

    for i=1:length(imgFiles)

        testName = imgFiles(i).name;

        img = im2double(imread(fullfile(testDir, testName)));

        if size(img, 3)==3

            img = rgb2gray(img);

        end

        img = imresize(img, [targetSize, targetSize]);

        imgFeatures = visualFeatures(img, patchSize, stepSize);

        

        [fN, ~] = size(imgFeatures);

        

        bow = zeros(1,k);   % Initialization word bag vector

        for j=1:fN

            diffMat = repmat(imgFeatures(j,:), [k, 1]) - C;

            distanceMat = sqrt(sum(diffMat.^2,2));  % Calculate the distance between the test image and the k centroids

            [~,ii] = min(distanceMat);  % Select the index of the centroid with the smallest distance

            bow(1, ii) = bow(1,ii) + 1; % Distance minimum centroid frequency +1

        end

        testBoWs = [testBoWs;bow];

        

        if isempty(testNames)

            testNames = testName;

        else

            testNames = char(testNames, testName);

        end

        

        if mod(i, 200) == 0

            fprintf('Done %d/%d\n', i, length(imgFiles));

        end

    end

    save run2_testBoWs_testNames.mat testBoWs testNames;

else

    disp('Please load .mat file!');

end

end

 

% extract the features of every image using small patches

function imgFeatures = visualFeatures(img, patchSize, stepSize)

[imgRow, imgCol] = size(img);

startY = 1;

imgFeatures = [];

while (startY + patchSize-1) <= imgRow % Move the patch along the y-axis and the x-axis of the picture, the data of the patch is the extracted features

    startX = 1;

    while (startX + patchSize-1) <= imgCol

        patch = img(startY:startY+patchSize-1, startX:startX+patchSize-1);

        [pR, pC] = size(patch);

        patch = reshape(patch, 1, pR*pC);

        % Convert the extracted features into vectors, and make them zero

        % mean and unit length at the same time

        patch = patch - mean(patch);

        patch = patch / norm(patch);

        imgFeatures = [imgFeatures; patch];

        

        startX = startX + stepSize;

    end

    startY = startY + stepSize;

end

end


