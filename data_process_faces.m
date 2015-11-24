% Data analysis practice on yales faces data 
% Datasetm: Yale Face Database
%
% Author: Xuanpei Ouyang

% pca by using svd
load('Face_40by40_500.mat')
MeanFace = mean(facemat,2);
Z = facemat - repmat(MeanFace, [1, size(facemat,2)]);
[U, S, V] = svd(Z);
Proj = U(:,1:3)'*Z;
Proj = Proj';

centers = datasample(Proj, 4);  % let k = 4 and get random initial cluster center

for i=1:10
    
    dist = zeros(500,3);
    
    for j=1:size(centers,1)
        dist(:,j) = sqrt((Proj(:,1) - centers(j,1)).^2 + (Proj(:,2) - centers(j,2)).^2);
    end

    [v, g_ind] = min(dist, [], 2);

    figure();
    scatter3(Proj(g_ind == 1,1), Proj(g_ind == 1,2), Proj(g_ind == 1,3),20,'r','filled');
    hold on;
    scatter3(centers(1,1),centers(1,2),centers(1,3),80,'Marker','x','MarkerEdgeColor',[0.6 0 0],'LineWidth',4);
    scatter3(Proj(g_ind == 2,1), Proj(g_ind == 2,2),Proj(g_ind == 2,3),20,'g','filled');
    scatter3(centers(2,1),centers(2,2),centers(2,3),80,'Marker','x','MarkerEdgeColor',[0 0.6 0],'LineWidth',4);
    scatter3(Proj(g_ind == 3,1), Proj(g_ind == 3,2),Proj(g_ind == 3,3),20,'b','filled');
    scatter3(centers(3,1),centers(3,2),centers(3,3),80,'Marker','x','MarkerEdgeColor',[0 0 0.6],'LineWidth',4);
    scatter3(Proj(g_ind == 4,1), Proj(g_ind == 4,2),Proj(g_ind == 4,3),20,'y','filled');
    scatter3(centers(4,1),centers(4,2),centers(4,3),80,'Marker','x','MarkerEdgeColor',[0.6 0.6 0],'LineWidth',4);
    hold off;
    
    centers(1,:) = mean(Proj(g_ind == 1, :));
    centers(2,:) = mean(Proj(g_ind == 2, :));
    centers(3,:) = mean(Proj(g_ind == 3, :));
    centers(4,:) = mean(Proj(g_ind == 4, :));
    
    % use the for loop to repeat this for ten times
    title(sprintf('Iteration %d',i));
end
