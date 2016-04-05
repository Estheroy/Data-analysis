% One-Hot Encoding
% Given a n x 1 vector as original labels and convert original labels into
% binary code labels by using one-hot encoding.
% 
% Input: labels: a n by 1 vector
%       
% Output: new_labels: the one-hot encoding of original labels
%         
% Author: Xuanpei Ouyang
function new_labels = OneHotEncoding(labels)
    
    numOfUniqueLabel = length(unique(labels));
    sortedUniqueLabel = unique(labels);
    new_labels = zeros(length(labels), numOfUniqueLabel);
   
    for i = 1:length(labels)
        new_labels(i,:) = (labels(i) == sortedUniqueLabel);
    end
end
