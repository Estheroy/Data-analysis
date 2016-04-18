% Implement k-th nearest neighbors classifier
% 
% Input: X: a matrix for training set features 
%        Y_NUM: a vector raining set class labels
%        toTest: a vector for testing set features
%        kVal: specify the k value in k-th nearest neighbors
%
% Return: result: a vector for predicted class labels given features in toTest
% 
% Author: Xuanpei Ouyang
function[result] = KNN(X,Y_NUM,toTest,kVal)
    result = zeros(size(toTest,1),1);

    for iter=1:size(toTest,1)
        euclidian_dist = zeros(size(Y_NUM,1),1);
        for j=1:size(X,2)
            euclidian_dist = euclidian_dist + (X(:,j) - toTest(iter,j)).^2;
        end
        euclidian_dist = sqrt(euclidian_dist);

        topNMinIndex = zeros(kVal,1);
        for i=1:kVal
            [~, minIndex] = min(euclidian_dist);
            topNMinIndex(i) = minIndex;
            euclidian_dist(minIndex) = max(euclidian_dist);
        end

        Classification_Result = Y_NUM(topNMinIndex);
        countForZero = 0;
        countForOne = 0;
        
        for i=1:size(Classification_Result,1)
            if(double(Classification_Result(i)) == 0)
                countForZero = countForZero+1;
            else
                countForOne = countForOne+1;
            end
        end
        
        if(countForZero > countForOne)
            result(iter) = 0;
        else
            result(iter) = 1;
        end
    end    
end
