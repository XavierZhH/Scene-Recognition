%Before runing this codes, install resnet50 please

% This part is revised from
% https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html

rootFolder = 'C:\Users\hz2m18\Downloads\training'

imds = imageDatastore(fullfile(rootFolder),'IncludeSubfolders', true, 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

% Load pretrained network
net = resnet50();

% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, imds, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = imds.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


rootFolder_test = 'C:\Users\hz2m18\Downloads\testing';
imds_test = dir(fullfile(rootFolder_test,'*.jpg'));
imds_test = sortObj(imds_test);
imds_test_name = {imds_test.name};


fp = fopen('run3.txt', 'w');
for i = 1: length(imds_test_name)
    % Extract test features using the CNN
    imgtest = imread(fullfile(rootFolder_test, imds_test_name{i}));
    augmentedTestSet = augmentedImageDatastore(imageSize, imgtest, 'ColorPreprocessing', 'gray2rgb');
    testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

    % Pass CNN image features to trained classifier
    predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
    fprintf(fp, '%s %s \r\n', char(imds_test_name{i}), char(predictedLabels));
end
fclose(fp);
    