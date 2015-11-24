% Implement K means clustering with visual representatiaon
% Compute the likelihood function according to Euclidian distance
%
% Author: Xuanpei Ouyang

clear all;
load('kmeandata.mat');
K = [4,15,20,50];
figure(1);
for i = 1:length(K)
    %initialize variables
    tol = 0;
    dist = [];
    diff = 0;
    % generate K number of samples from dataset
    centers = kmeandata(randi(size(kmeandata,1),K(i),1),:);
    for t=1:200
        % compute the distance
        for j=1:K(i)
            dist(:,j) = sqrt((kmeandata(:,1) - centers(j,1)).^2 + (kmeandata(:,2) - centers(j,2)).^2);
        end
        % assign each sample to its closest cluster center
        [v, g_ind] = min(dist, [], 2);      
        % compute the total distance
        tol = sum(v.^2);
        prev_centers = centers;
        % update the cluster centers
        for z = 1:K(i)
            centers(z,:) = mean(kmeandata(g_ind == z, :));
        end  
        % detect the amount of shift in cluster centers between the current iteration and the previous iteration
        diff = sum(abs(prev_centers - centers));
    end
    subplot(2,2,i);drawnow; 
    for m = 1:K(i);
        colorspec = rand(1,3);
        scatter(kmeandata(g_ind == m,1),kmeandata(g_ind == m,2),20,colorspec,'filled');
        hold on;
        scatter(centers(m,1),centers(m,2),80,'Marker','x','MarkerEdgeColor',[0.6 0.6 0.6],'LineWidth',4);
    end
    hold off;
end

figure();
plot(K,nlglvec);

% plot with prior1
figure();
subplot(1,3,1);
plot(K,exp(-nlglvec));
subplot(1,3,2);
plot(K,prior1);
subplot(1,3,3);
plot(K,exp(-nlglvec).*prior1);

% plot with prior2
figure();
subplot(1,3,1);
plot(K,exp(-nlglvec));
subplot(1,3,2);
plot(K,prior2);
subplot(1,3,3);
plot(K,exp(-nlglvec).*prior2);

% plot with prior3
figure();
subplot(1,3,1);
plot(K,exp(-nlglvec));
subplot(1,3,2);
plot(K,prior3);
subplot(1,3,3);
plot(K,exp(-nlglvec).*prior3);