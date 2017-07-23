%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Stuart Daries, Nitish Mutha
% Student ID: 16079408 ,15113106
% Question: 9a
% Section: Part 1
% Description: Baseline versus full linear regression.
% ------------------------------------------------------------------%

close all;
clear all;
addpath('../library')

load('boston.mat')

%% Ex9(a)

% 20 trails
for j=1:20
    
    data_train = datasample(boston,round(2/3*size(boston,1)),1,'Replace',false);
    % Stored left over rows in the test set 1/3
    data_test = setdiff(boston, data_train, 'rows');

    x_train_v = ones(round(2/3*size(boston,1)),1);
    y_train = data_train(:,14);

    x_test_v = ones(round(1/3*size(boston,1)),1);
    y_test = data_test(:,14);   

    % Calculate weight vector w
    w_star = learnModel(x_train_v, y_train);
    % Calculate MSE
    mse_train(j) = meanSquareError(w_star,x_train_v,y_train,size(x_train_v,1));
    mse_test(j) = meanSquareError(w_star,x_test_v,y_test,size(x_test_v,1));

end
% Calculate mean and standard deviation of trian and test set.
avg_mse_train = mean(mse_train)
std_train = std(mse_train)

avg_mse_test = mean(mse_test)
std_test = std(mse_test)

mse = [avg_mse_train; avg_mse_test]';

