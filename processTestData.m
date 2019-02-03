% Processing test data for prediction

function processTestData(testDir, targetSize, load)

if ~load

    testImgs = [];

    testNames = [];

    imgFiles = dir(fullfile(testDir, '*.jpg')); % Get all jpg images under the test image path
    imgFiles = sortObj(imgFiles); % Get all test images by natural-order

    for i=1:length(imgFiles)    % Traverse all test images for processing,2988

        testName = imgFiles(i).name;

        img = im2double(imread(fullfile(testDir,testName))); % convert a test image to a pixel matrix 

        if size(img, 3)==3

            img = rgb2gray(img);

        end

        % make the test image the same size as the training image

        img = imresize(img, [targetSize, targetSize]);

        img = reshape(img, 1, targetSize*targetSize);

        img = img - mean(img);  %zero mean

        img = img / norm(img);  %unit length

        testImgs = [testImgs;img];

        if isempty(testNames)

            testNames = testName;

        else

            testNames = char(testNames, testName);

        end

    end

    save run1_testImgs_testNames.mat testImgs testNames;    % save test data matrix and test data filenames to a .mat file

else

    disp('You can load .mat file now!');

end

end
