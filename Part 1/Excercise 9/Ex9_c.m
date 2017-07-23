%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Stuart Daries, Nitish Mutha
% Student ID: 16079408 ,15113106
% Question: 9c
% Section: Part 1
% Description: Baseline versus full linear regression.
% ------------------------------------------------------------------%


close all;
clear all;
addpath('../library')

load('boston.mat')

%% Ex9(c)

% 20 trials
for j=1:20
    
    % Radomly withdraw 2/3 data from Boston for training samples
    data_train = datasample(boston,round(2/3*size(boston,1)),1,'Replace',false);
    % Stored left over rows in the test set 1/3
    data_test = setdiff(boston, data_train, 'rows');
    % Segment the columns for training attributes
    x_train = data_train(:,1:13);
    % Take the last column as the true value we are trying to predict
    y_train = data_train(:,14);
    
    x_test = data_test(:,1:13);
    y_test = data_test(:,14);
    
    % Train the model vector w
    x_train_v = [x_train, ones(size(x_train,1),1)];
    w_star = learnModel(x_train_v, y_train);
    mse_train(j) = meanSquareError(w_star,x_train_v,y_train,size(x_train_v,1));
    
    % Test the learnt vector w
    x_test_v = [x_test, ones(size(x_test,1),1)];
    mse_test(j) = meanSquareError(w_star,x_test_v,y_test,size(x_test_v,1));
    
end

avg_mse_train = mean(mse_train)
std_train = std(mse_train)

avg_mse_test = mean(mse_test)
std_test = std(mse_test)

mse = [avg_mse_train; avg_mse_test]';

%% plots
figure;
boxplot(mse_train);
set(gcf, 'Color', 'w');
% xlabel('Attributes','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('Train','Test','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex9c_mse_box_train','-depsc');
close all;

figure;
boxplot(mse_test);
set(gcf, 'Color', 'w');
% xlabel('Attributes','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('Train','Test','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex9c_mse_box_test','-depsc');
close all;
