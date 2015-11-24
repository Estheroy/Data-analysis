% Implement PCA by function svd
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
figure;

Z = facemat - repmat(MeanFace, [1, size(facemat,2)]);
tic;
[U, S, V] = svd(Z);
for i=1:25
    tstart = tic;
    eigenface = reshape(U(:,i),40,40);
    subplot(5,5,i);
    imshow(eigenface,[]);
end
time_elapsed = toc(tstart)

