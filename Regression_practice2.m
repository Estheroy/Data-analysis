% Multiple linear regression and error analysis for predicting news popularity
% Dataset download from UCI Machine Learning Repository
%
% Author: Xuanpei Ouyang

clear;
load('OnlineNewsPopularity');

A1 = cat(2,ones(length(Pub_Weekdays),1),Pub_Weekdays);
W1 = A1\Popularity;

A2 = cat(2,ones(length(Content),1),Content);
W2 = A2\Popularity;

A3 = cat(2,ones(length(Stats),1),Stats);
W3 = A3\Popularity;

subplot(1,3,1);
scatter(Popularity,A1*W1,20,'b','filled');

subplot(1,3,2);
scatter(Popularity,A2*W2,20,'b','filled');

subplot(1,3,3);
scatter(Popularity,A3*W3,20,'b','filled');

A1test = cat(2,ones(length(Pub_Weekdays),1),Pub_Weekdays);
Etest1 = mean((Popularity - A1test*W1).^2)

A2test = cat(2, ones(length(Content),1),Content);
Etest2 = mean((Popularity - A2test*W2).^2)
 
A3test = cat(2, ones(length(Stats),1),Stats);
Etest3 = mean((Popularity - A3test*W3).^2)