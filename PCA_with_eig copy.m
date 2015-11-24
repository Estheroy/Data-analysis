% Implement PCA by function eig
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

MeanFace = mean(facemat,2);
mean_face = reshape(MeanFace,40,40);
figure;
imshow(mean_face,[]);

Z = facemat - repmat(MeanFace, [1, size(facemat,2)]);
C = Z*Z'/size(facemat,2);
tic;
[V, D] = eig(C);
[sv si] = sort(diag(D),'descend');
Vs = V(:,si);
figure;
for i=1:25
    tstart = tic;
    Vs_reshape = reshape(Vs(:,i),40,40);
    subplot(5,5,i);
    imshow(Vs_reshape,[]);
end
time_elapsed = toc(tstart)

figure;
Proj = Vs(:,1:3)'*Z;
scatter3(Proj(1,:), Proj(2,:), Proj(3,:), 20, 'filled');

figure;
for i=1:5
    subplot(1,5,i);
    ReFace = Vs(:,1:i*20) * Vs(:,1:i*20)' * Z(:,1) + MeanFace;
    ReFace = reshape(ReFace, 40, 40);
    imshow(ReFace,[]);
end
