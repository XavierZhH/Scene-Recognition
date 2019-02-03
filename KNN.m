% Prediction of test images using KNN

function resultLabel = KNN(input,data,labels,k)

[datarow , ~] = size(data);

% +++++adding compare with O-distance(increasing)

distanceMat = cosineDistance(repmat(input,[datarow,1]), data);  % Calculate the cosine similarity of a given test image to all training images(descend)

weightMat = distanceMat.^2; % Voting weight, square of similarity

[B , IX] = sort(distanceMat,'descend'); % Sort by cosine similarity in descending order

len = min(k,length(B));

indexes = IX(1:len);    % Get k indexes with the highest similarity value

map = containers.Map('KeyType','double','ValueType','double');

 

% Get the categories to which the k most similar training images belong, and each time these categories occur, multiply by weight to vote

for j=1:length(indexes)

    label = labels(indexes(j));

    weight = weightMat(indexes(j));

    if map.isKey(label) % vote by weight

        map(label) = map(label) + 1*weight;

    else

        map(label) = 1*weight;

    end

end

[~,idx] = max(cell2mat(map.values()));  % The category with the most votes is considered to be the category of the test image.

keyMat = cell2mat(map.keys());

resultLabel = keyMat(idx);

end