% Implement mean means clustering with visual representation
%
% Author: Xuanpei Ouyang

load('clustering_data');
center = [4,4];
radius = 1.5;

figure()
scatter(kmeandata(:,1),kmeandata(:,2),20,'b','filled');
hold on;

for i=1:10
    
    plot(center(:,1) + radius*cos([0:0.01:2*pi]),center(:,2) + radius*sin([0:0.01:2*pi]), ...
    '--','Color',[0,1,0],'LineWidth',1);

    dist(:,1) = sqrt((kmeandata(:,1) - center(1,1)).^2 + (kmeandata(:,2) - center(1,2)).^2);
    ind = dist <= radius;
    center = mean(kmeandata(ind,1:2));
end

plot(center(:,1),center(:,2),'r*');

hold off;
