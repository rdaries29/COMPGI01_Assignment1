%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Stuart Daries, Nitish Mutha
% Student ID: 16079408 ,15113106
% Question: 5
% Section: Part 1
% Description: Tuning Regularization Parameter with Validation Set
% ------------------------------------------------------------------%

%%
clear all
close all
clc

addpath('../library')

%% Ex5(a) and Ex5(b)

gamma_power = -6:1:3;
gamma = 10.^gamma_power;

result_mse_train_100 = [];
result_mse_train_10 = [];

result_mse_valid_100 = [];
result_mse_valid_10 = [];

result_mse_test_100 = [];
result_mse_test_10 = [];

result_w_100 = [];
result_w_10 = [];

best_gamma_100 = zeros(200,1);
best_gamma_10 = zeros(200,1);

g_average_mse_test_100 = [];
g_average_mse_test_10 = [];

for j = 1:200
    
    w_j = randn(10,1);
    x_j = randn(600,10);
    n_j = randn(600,1);
    w_100 = [];
    w_10 = [];
    
    mse_train_100 = [];
    mse_valid_100 = [];
    mse_test_100 = [];
    
    mse_train_10 = [];
    mse_valid_10 = [];
    mse_test_10 =[];
    

    for i=1: size(gamma,2)
        % Calculate MSE for each data set with 100 training samples
        [result_w_100_j, result_mse_train_100_j,result_mse_valid_100_j, result_mse_test_100_j] = calculateLSRex5(x_j,w_j,n_j,80,20,500,gamma(i));
        w_100 = [w_100; result_w_100_j];
        mse_train_100 = [mse_train_100, result_mse_train_100_j];
        mse_valid_100 = [mse_valid_100, result_mse_valid_100_j];
        mse_test_100 = [mse_test_100, result_mse_test_100_j];
        % Calculate MSE for each data set with 10 training samples
        [result_w_10_j, result_mse_train_10_j,result_mse_valid_10_j, result_mse_test_10_j] = calculateLSRex5(x_j,w_j,n_j,8,2,500,gamma(i));
        w_10 = [w_10; result_w_10_j];
        mse_train_10 = [mse_train_10, result_mse_train_10_j];
        mse_valid_10 = [mse_valid_10, result_mse_valid_10_j];
        mse_test_10 = [mse_test_10, result_mse_test_10_j];
        
    end
    % Find optimal gamma value from validation set MSE
    [min_mse_100,idx_min_100] = min(mse_valid_100);
    best_gamma_100(j,1) = gamma(idx_min_100);
    
    [min_mse_10,idx_min_10] = min(mse_valid_10);
    best_gamma_10(j,1) = gamma(idx_min_10);
    
    result_w_100 = [result_w_100 ; w_100];
    result_w_10 = [result_w_10 ; w_10];
    
    result_mse_train_100 = [result_mse_train_100 ; mse_train_100];
    result_mse_train_10 = [result_mse_train_10 ; mse_train_10];
    
    result_mse_valid_100 = [result_mse_valid_100 ; mse_valid_100];
    result_mse_valid_10 = [result_mse_valid_10 ; mse_valid_10];
    
    result_mse_test_100 = [result_mse_test_100 ; mse_test_100];
    result_mse_test_10 = [result_mse_test_10 ; mse_test_10];
    
    [w_test_100, g_result_mse_test_100] = calculateLSRTest(x_j,w_j,n_j,80,20,500,gamma(idx_min_100));
    g_average_mse_test_100 = [g_average_mse_test_100, g_result_mse_test_100];
    
    [w_test_10, g_result_mse_test_10] = calculateLSRTest(x_j,w_j,n_j,8,2,500,gamma(idx_min_10));
    g_average_mse_test_10 = [g_average_mse_test_10, g_result_mse_test_10];
    
    
end

% Compute averages
average_mse_train_100 = mean(result_mse_train_100);
average_mse_valid_100 = mean(result_mse_valid_100);
average_mse_test_100 = mean(result_mse_test_100);

average_mse_train_10 = mean(result_mse_train_10);
average_mse_valid_10 = mean(result_mse_valid_10);
average_mse_test_10 = mean(result_mse_test_10);

% plots for mse vs gamma values for 100
figure;
plot(log(gamma),average_mse_train_100,'r-*');
hold on;
plot(log(gamma),average_mse_valid_100,'g-o');
plot(log(gamma),average_mse_test_100,'b-+');
grid on;
set(gcf, 'Color', 'w');
xlabel('Regularization parameter (\gamma)','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{train}','MSE_{validation}','MSE_{test}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex5a_80_20','-depsc');
close all;

% plots for mse vs gamma values for 10
figure;
plot(log(gamma),average_mse_train_10,'r-*');
hold on;
plot(log(gamma),average_mse_valid_10,'g-o');
plot(log(gamma),average_mse_test_10,'b-+');
grid on;
set(gcf, 'Color', 'w');
xlabel('Regularization parameter (\gamma)','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{train}','MSE_{validation}','MSE_{test}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex5b_8_2','-depsc');
close all;

% plots for test mse vs experiments for 100 and 10
figure;
plot(g_average_mse_test_100,'r*');
hold on;
plot(g_average_mse_test_10,'b+');
grid on;
set(gcf, 'Color', 'w');
xlabel('Experiments','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{100}','MSE_{10}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
ylim([0.0 40])
print('ex5b_mse_optimal_gamma','-depsc');
close all;

%% Ex5(c)
mean_gamma_100 = mean(best_gamma_100)
mean_gamma_10 = mean(best_gamma_10)

%% Ex5(d)

% Repeating tuning of regularization parameter with validation set with
% data similair to Ex1(a)-(d)
result_mse_train_100 = [];
result_mse_train_10 = [];

result_mse_valid_100 = [];
result_mse_valid_10 = [];

result_mse_test_100 = [];
result_mse_test_10 = [];

result_w_100 = [];
result_w_10 = [];

best_gamma_100 = zeros(200,1);
best_gamma_10 = zeros(200,1);

g_average_mse_test_100 = [];
g_average_mse_test_10 = [];

for j = 1:200

    w_j = randn(1,1);
    x_j = randn(600,1);
    n_j = randn(600,1);
    w_100 = [];
    w_10 = [];
    
    mse_train_100 = [];
    mse_valid_100 = [];
    mse_test_100 = [];
    
    mse_train_10 = [];
    mse_valid_10 = [];
    mse_test_10 =[];
    

    for i=1: size(gamma,2)
        % Calculate MSE for each data set with 100 training samples
        [result_w_100_j, result_mse_train_100_j,result_mse_valid_100_j, result_mse_test_100_j] = calculateLSRex5(x_j,w_j,n_j,80,20,500,gamma(i));
        w_100 = [w_100; result_w_100_j];
        mse_train_100 = [mse_train_100, result_mse_train_100_j];
        mse_valid_100 = [mse_valid_100, result_mse_valid_100_j];
        mse_test_100 = [mse_test_100, result_mse_test_100_j];
        % Calculate MSE for each data set with 10 training samples
        [result_w_10_j, result_mse_train_10_j,result_mse_valid_10_j, result_mse_test_10_j] = calculateLSRex5(x_j,w_j,n_j,8,2,500,gamma(i));
        w_10 = [w_10; result_w_10_j];
        mse_train_10 = [mse_train_10, result_mse_train_10_j];
        mse_valid_10 = [mse_valid_10, result_mse_valid_10_j];
        mse_test_10 = [mse_test_10, result_mse_test_10_j];
        
    end
    % Find optimal gamma value from validation set MSE
    [min_mse_100,idx_min_100] = min(mse_valid_100);
    best_gamma_100(j,1) = gamma(idx_min_100);
    
    [min_mse_10,idx_min_10] = min(mse_valid_10);
    best_gamma_10(j,1) = gamma(idx_min_10);
    
    result_w_100 = [result_w_100 ; w_100];
    result_w_10 = [result_w_10 ; w_10];
    
    result_mse_train_100 = [result_mse_train_100 ; mse_train_100];
    result_mse_train_10 = [result_mse_train_10 ; mse_train_10];
    
    result_mse_valid_100 = [result_mse_valid_100 ; mse_valid_100];
    result_mse_valid_10 = [result_mse_valid_10 ; mse_valid_10];
    
    result_mse_test_100 = [result_mse_test_100 ; mse_test_100];
    result_mse_test_10 = [result_mse_test_10 ; mse_test_10];
    
    [w_test_100, g_result_mse_test_100] = calculateLSRTest(x_j,w_j,n_j,80,20,500,gamma(idx_min_100));
    g_average_mse_test_100 = [g_average_mse_test_100, g_result_mse_test_100];
    
    [w_test_10, g_result_mse_test_10] = calculateLSRTest(x_j,w_j,n_j,8,2,500,gamma(idx_min_10));
    g_average_mse_test_10 = [g_average_mse_test_10, g_result_mse_test_10];
    
    
end

% Compute averages
average_mse_train_100 = mean(result_mse_train_100);
average_mse_valid_100 = mean(result_mse_valid_100);
average_mse_test_100 = mean(result_mse_test_100);

average_mse_train_10 = mean(result_mse_train_10);
average_mse_valid_10 = mean(result_mse_valid_10);
average_mse_test_10 = mean(result_mse_test_10);

% plots for mse vs gamma values for 100
figure;
plot(log(gamma),average_mse_train_100,'r-*');
hold on;
plot(log(gamma),average_mse_valid_100,'g-o');
plot(log(gamma),average_mse_test_100,'b-+');
grid on;
set(gcf, 'Color', 'w');
xlabel('Regularization parameter (\gamma)','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{train}','MSE_{validation}','MSE_{test}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex5d_80_20','-depsc');
close all;

% plots for mse vs gamma values for 10
figure;
plot(log(gamma),average_mse_train_10,'r-*');
hold on;
plot(log(gamma),average_mse_valid_10,'g-o');
plot(log(gamma),average_mse_test_10,'b-+');
grid on;
set(gcf, 'Color', 'w');
xlabel('Regularization parameter (\gamma)','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{train}','MSE_{validation}','MSE_{test}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex5d_8_2','-depsc');
close all;



