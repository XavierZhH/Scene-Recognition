% Calculating cosine similarity

function cosineSimilarity = cosineDistance(matA,matB)

s = sum(matA.*matB, 2);

m = (sum(abs(matA).^2,2).^(1/2)) .*(sum(abs(matB).^2,2).^(1/2));

cosineSimilarity = s./m;

end