% Implement Perceptron with visual representation 
%
% Author: Xuanpei Ouyang

clear;
load('two_cluster_data');

b = -3;
w1 = 2;
w2 = 1;

figure();

while 1 > 0
    err_id=[];
    output = []; %%%%
    incorrect = [];
    N = size(target,1);
    for i = 1:N %loop through all points
        net=w1*x1(i)+w2*x2(i)+b;
        if net>=0 %set output to 1 if net >=0
            output(i) = 1;
        else %set output to -1 if net <0
            output(i) = -1; 
        end
    
        if output(i)==target(i) 
            incorrect(i) = 0;
        else
            incorrect(i) = 1;
            err_id=[err_id i]; %add index of index of incorrect output to err_id
        end
    end
    
    indicator = any(err_id);
    
    if (sum(indicator) == 0)
        scatter(x1(target==-1),x2(target==-1),10,'g','filled');
        hold on
        scatter(x1(target==1),x2(target==1),10,'r','filled');
        x_test = -11:11; % define an arbitrary x sequence for drawing the line 
        y_test = (-w1*x_test-b)/w2;
        plot(x_test,y_test,'k','linewidth',2);
        
        if any(err_id)
            scatter(x1(target==-1),x2(target==-1),10,'g','filled');
            hold on
            scatter(x1(target==1),x2(target==1),10,'r','filled');
            x_test = -11:11; % define an arbitrary x sequence for drawing the line 
            y_test = (-w1*x_test-b)/w2;
            plot(x_test,y_test,'k','linewidth',2);
            scatter(x1(err_id),x2(err_id) ,50,'k','linewidth',2);
        end
        break;
    end
    
    w1=w1+(target(err_id(1))-output(err_id(1)))*x1(err_id(1)); 
    w2=w2+(target(err_id(1))-output(err_id(1)))*x2(err_id(1)); 
    b = b+(target(err_id(1))-output(err_id(1)));
    
end
