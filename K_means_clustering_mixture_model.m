% Implement K means clustering
% Compute the likelihood function according to mixture model
%
% Author: Xuanpei Ouyang

clear all;
load('kmeandata.mat');
K = [1:1:50];
nlglvec = [];
for iter = 1:10
    
    %initialize variables
    temp = [];
    dist = [];
    diff = 0;
    pik = [];
    pik_long = zeros(1,size(kmeandata,1));
    for i = 1:length(K);
        centers = kmeandata(randi(size(kmeandata,1),K(i),1),:);
        for t=1:200
            for j=1:K(i)
                dist(:,j) = sqrt((kmeandata(:,1) - centers(j,1)).^2 + (kmeandata(:,2) - centers(j,2)).^2);
            end
            [v, g_ind] = min(dist, [], 2);  
            % get the pik
            for z = 1:K(i)
                pik(z) = sum(g_ind == z); % count the occurence of each number in order
            end
            pik = pik./size(kmeandata,1); % get the pik
            
            for z = 1:size(kmeandata,1)
                pik_long(z) = pik(g_ind(z));
            end
            
            temp(i) = sum((pik_long').*exp((v.^2))); % calculate the p(x|theta)based on the mixture model
            prev_centers = centers;
            for z = 1:K(i)
                centers(z,:) = mean(kmeandata(g_ind == z, :));
            end
            % detect the amount of shift in cluster centers between the current iteration and the previous iteration
            diff = sum(abs(prev_centers - centers));
        end 
    end
    nlglvec = [nlglvec; temp];
end

figure();
nlglvec_mean = mean(nlglvec);
plot(K,nlglvec_mean);

% plot with prior1
figure();
subplot(1,3,1);
plot(K,exp(-nlglvec_mean));
subplot(1,3,2);
plot(K,prior1);
subplot(1,3,3);
plot(K,exp(-nlglvec_mean').*prior1);

% plot with prior2
figure();
subplot(1,3,1);
plot(K,exp(-nlglvec_mean));
subplot(1,3,2);
plot(K,prior2);
subplot(1,3,3);
plot(K,exp(-nlglvec_mean').*prior2);

% plot with prior3
figure();
subplot(1,3,1);
plot(K,exp(-nlglvec_mean));
subplot(1,3,2);
plot(K,prior3);
subplot(1,3,3);
plot(K,exp(-nlglvec_mean').*prior3);