% Implement PCA by built in function pca
% Image process faces
% Dataset: Yale Face Database
%
% Author: Xuanpei Ouyang

load('Face_40by40_500.mat')

figure;
for i=1:25
    temp_face = reshape(facemat(:,i),40,40);
    subplot(5,5,i);
    imshow(temp_face,[]);
end

COEFF = pca(facemat', 'Algorithm', 'eig');
for i=1:25
    temp_COEFF = reshape(COEFF(:,i),40,40);
    subplot(5,5,i);
    imshow(temp_COEFF,[]);
end
