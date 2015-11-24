% Polynomial regression data fitting and error analysis practice
% linear Regression, quadratic regression, cubic regression
% Dataset download from UCI Machine Learning Repository
%
% Author: Xuanpei Ouyang

load('LinearRegression');

A1 = cat(2,ones(length(Xtrain),1),Xtrain); % cat(2,A,B) -> [A,B]
W1 = A1\Ytrain;  % A1 * Ytrain = W1  (W1: 2x1)

A2 = cat(2,ones(length(Xtrain),1),Xtrain,Xtrain.^2);
W2 = A2\Ytrain;  % A2 * Ytrain = W2  (W2: 3x1)

A3 = cat(2,ones(length(Xtrain),1),Xtrain,Xtrain.^2,Xtrain.^3);
W3 = A3\Ytrain;  % A3 * Ytrain = W3  (W3: 4x1)

Xpred = linspace(min(Xtrain),max(Xtrain),500)'; % create an X series to draw lines
A1pred = cat(2, ones(length(Xpred),1),Xpred);
A2pred = cat(2, ones(length(Xpred),1),Xpred,Xpred.^2);
A3pred = cat(2, ones(length(Xpred),1),Xpred,Xpred.^2,Xpred.^3);

Ypred1 = A1pred*W1; 
Ypred2 = A2pred*W2; 
Ypred3 = A3pred*W3; 

subplot(1,3,1); 
scatter(Xtrain,Ytrain,20,'b','filled'); 
hold on; 
plot(Xpred,Ypred1,'r','LineWidth',3);

subplot(1,3,2); 
scatter(Xtrain,Ytrain,20,'b','filled'); 
hold on; 
plot(Xpred,Ypred2,'r','LineWidth',3);

subplot(1,3,3); 
scatter(Xtrain,Ytrain,20,'b','filled');
hold on; 
plot(Xpred,Ypred3,'r','LineWidth',3);

Etrain1 = mean((Ytrain - A1*W1).^2);
Etrain2 = mean((Ytrain - A2*W2).^2);
Etrain3 = mean((Ytrain - A3*W3).^2);

A1test = cat(2,ones(length(Xtest),1),Xtest);
Etest1 = mean((Ytest - A1test*W1).^2);

A2test = cat(2, ones(length(Xtest),1),Xtest,Xtest.^2);
Etest2 = mean((Ytest - A2test*W2).^2);
 
A3test = cat(2, ones(length(Xtest),1),Xtest,Xtest.^2,Xtest.^3);
Etest3 = mean((Ytest - A3test*W3).^2);
