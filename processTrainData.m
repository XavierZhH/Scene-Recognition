% Processing training data, including resize, converting to vectors, zero

% mean and unit length, etc

function processTrainData(trainDir, targetSize, load)

if ~load

    listDir = dir(trainDir);

    fileNum=size(listDir, 1);%15row

    

    trainImgs = [];

    trainLabels = [];

    labelNames = [];

    for i=1:fileNum%15

        label = listDir(i).name;

        if ~contains(label, '.')

            disp(label);

            subDir = strcat(trainDir,'\',label);

            subList = dir(fullfile(subDir, '*.jpg'));

            for j=1:length(subList)

                img = im2double(imread(fullfile(subDir, subList(j).name))); % Read the image and convert the image into a pixel matrix

                if size(img, 3)==3  % Ignore the color channel and think of it as a gray figure.

                    img = rgb2gray(img);

                end

                img = imresize(img, [targetSize, targetSize]);  % Reduce the picture from around to the size

                img = reshape(img, 1, targetSize*targetSize); % Tile the pixel matrix into a vector(convert row)

                img = img - mean(img); % zero mean

                img = img / norm(img); % unit length

                trainImgs = [trainImgs;img];%1500images

                trainLabels = [trainLabels;i];%15classes

            end

        end

        if isempty(labelNames)

            labelNames = label;%loop 15

        else

            labelNames = char(labelNames, label);

        end

    end

    save run1_trainImgs_trainLabels.mat trainImgs trainLabels;  % Save the processed training dataset and training labels as a .mat file

    save run1_labelNames.mat labelNames; % Save the label names strings

else

    disp('You can load .mat file now!');

end

end
