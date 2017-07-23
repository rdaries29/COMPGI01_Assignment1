%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Stuart Daries, Nitish Mutha
% Student ID: 16079408 ,15113106
% Question: 6
% Section: Part 1
% Description: Tuning Regularization Parameter with Cross-Validation
% ------------------------------------------------------------------%

%%
clear all
close all
clc

addpath('../library')

%% Ex6(a) and Ex6(b)

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


g_average_mse_test_100 = [];
g_average_mse_test_10 = [];

w_j = randn(10,1);
x_j = randn(600,10);
n_j = randn(600,1);

% Perform 5 fold cross validation
for j = 1:5
    
    
    w_100 = [];
    w_10 = [];
    
    mse_train_100 = [];
    mse_valid_100 = [];
 
    
    mse_train_10 = [];
    mse_valid_10 = [];
    
    
    for i=1: size(gamma,2)
        [result_w_100_j, result_mse_train_100_j,result_mse_valid_100_j] = calculateLSRex6(x_j,w_j,n_j,100,gamma(i),j);
        w_100 = [w_100; result_w_100_j];
        mse_train_100 = [mse_train_100, result_mse_train_100_j];
        mse_valid_100 = [mse_valid_100, result_mse_valid_100_j];
        
        [result_w_10_j, result_mse_train_10_j,result_mse_valid_10_j] = calculateLSRex6(x_j,w_j,n_j,10,gamma(i),j);
        w_10 = [w_10; result_w_10_j];
        mse_train_10 = [mse_train_10, result_mse_train_10_j];
        mse_valid_10 = [mse_valid_10, result_mse_valid_10_j];
        
    end
    %Identify gamma that minimises MSE
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
     
end

mean_gamma_100 = mean(best_gamma_100)
mean_gamma_10 = mean(best_gamma_10)

% Average all dataset errors
average_mse_train_100 = mean(result_mse_train_100);
average_mse_valid_100 = mean(result_mse_valid_100);

average_mse_train_10 = mean(result_mse_train_10);
average_mse_valid_10 = mean(result_mse_valid_10);

mse_test_100 = [];
mse_test_10 =[];
% Calculate MSE for test set
for i=1: size(gamma,2)
    [result_w_100_j, result_mse_train_100_j,result_mse_test_100_j] = calculateLSRex4(x_j,w_j,n_j,100,500,gamma(i));
    mse_test_100 = [mse_test_100, result_mse_test_100_j];
    
    [result_w_10_j, result_mse_train_10_j,result_mse_test_10_j] = calculateLSRex4(x_j,w_j,n_j,10,500,gamma(i));
    mse_test_10 = [mse_test_10, result_mse_test_10_j];
    
end

% plots for mse vs gamma values for 100
figure;
plot(log(gamma),average_mse_train_100,'r-*');
hold on;
plot(log(gamma),average_mse_valid_100,'b-o');
plot(log(gamma),mse_test_100,'g-+');
grid on;
set(gcf, 'Color', 'w');
xlabel('log (\gamma)','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{avg train}','MSE_{avg cross-val}','MSE_{test}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex6_cv_100','-depsc');
close all;

% plots for mse vs gamma values for 10
figure;
plot(log(gamma),average_mse_train_10,'r-*');
hold on;
plot(log(gamma),average_mse_valid_10,'b-o');
plot(log(gamma),mse_test_10,'g-+');
grid on;
set(gcf, 'Color', 'w');
xlabel('log (\gamma)','FontSize',12);
ylabel('Mean Square Error','FontSize',12);
leg=legend('MSE_{avg train}','MSE_{avg cross-val}','MSE_{test}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex6_cv_10','-depsc');
close all;
