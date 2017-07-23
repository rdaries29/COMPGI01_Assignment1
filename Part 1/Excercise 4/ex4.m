%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Stuart Daries, Nitish Mutha
% Student ID: 16079408 ,15113106
% Question: 4
% Section: Part 1
% Description: Effect of Regularisation Parameter
% ------------------------------------------------------------------%

%%
clear all
close all
clc

addpath('../library')

%% Ex4(a)
w = randn(10,1);
x = randn(600,10);
n = randn(600,1);
gamma_power = -6:1:3;
% Range of gamma values
gamma = 10.^gamma_power;

%Initialise variables
result_mse_train_100 = [];
result_mse_test_100 = [];
for i = 1 : size(gamma,2)
    [w_estimator_100, mse_train_100, mse_test_100] = calculateLSRex4(x,w,n,100,500,gamma(i));
    result_mse_train_100 = [result_mse_train_100 ; mse_train_100];
    result_mse_test_100 = [result_mse_test_100 ; mse_test_100];
end

figure
plot(log(gamma),result_mse_train_100,'b-o');
hold on;
plot(log(gamma),result_mse_test_100,'r-*');
grid on;
set(gcf, 'Color', 'w');
xlabel('{log(\gamma)}','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE-train_{100}','MSE-test_{100}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex4a_mse_100','-depsc');
close all;


%% Ex4(b)
result_mse_train_10 = [];
result_mse_test_10 = [];
for i = 1 : size(gamma,2)
    [w_estimator_10, mse_train_10, mse_test_10] = calculateLSRex4(x,w,n,10,500,gamma(i));
    result_mse_train_10 = [result_mse_train_10 ; mse_train_10];
    result_mse_test_10 = [result_mse_test_10 ; mse_test_10];
end

figure
plot(log(gamma),result_mse_train_10,'b-o');
hold on;
plot(log(gamma),result_mse_test_10,'r-*');
grid on;
set(gcf, 'Color', 'w');
xlabel('{log(\gamma)}','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE-train_{10}','MSE-test_{10}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex4b_mse_10','-depsc');
close all;


%% Ex4(c)

result_mse_train_100 = [];
result_mse_train_10 = [];
result_mse_test_100 = [];
result_mse_test_10 = [];
result_w_100 = [];
result_w_10 = [];

% 200 independant trials
for j = 1:200
    
    w_j = randn(10,1);
    x_j = randn(600,10);
    n_j = randn(600,1);
    w_100 = [];
    w_10 = [];
    
    mse_train_100 = [];
    mse_train_10 = [];
    mse_test_100 = [];
    mse_test_10 =[];
    
    % Varying regularisation parameter
    for i=1: size(gamma,2)
        [result_w_100_j, result_mse_train_100_j, result_mse_test_100_j] = calculateLSRex4(x_j,w_j,n_j,100,500,gamma(i));
        w_100 = [w_100, result_w_100_j];
        mse_train_100 = [mse_train_100, result_mse_train_100_j];
        mse_test_100 = [mse_test_100, result_mse_test_100_j];
        
        [result_w_10_j, result_mse_train_10_j, result_mse_test_10_j] = calculateLSRex4(x_j,w_j,n_j,10,500,gamma(i));
        w_10 = [w_10, result_w_10_j];
        mse_train_10 = [mse_train_10, result_mse_train_10_j];
        mse_test_10 = [mse_test_10, result_mse_test_10_j];
        
    end
    result_mse_train_100 = [result_mse_train_100 ; mse_train_100];
    result_mse_train_10 = [result_mse_train_10 ; mse_train_10];
    
    result_mse_test_100 = [result_mse_test_100 ; mse_test_100];
    result_mse_test_10 = [result_mse_test_10 ; mse_test_10];
    
    result_w_100 = [result_w_100 ; w_100];
    result_w_10 = [result_w_10 ; w_10];
    
end

% Computing the average result
average_mse_train_100 = mean(result_mse_train_100);
average_mse_test_100 = mean(result_mse_test_100);
average_mse_train_10 = mean(result_mse_train_10);
average_mse_test_10 = mean(result_mse_test_10);


figure
plot(log(gamma),average_mse_train_100,'b-*');
hold on;
grid on;
plot(log(gamma),average_mse_test_100,'r-o');
set(gcf, 'Color', 'w');
xlabel('{log(\gamma)}','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE-train_{100}','MSE-test_{100}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex4c_mse_200_100','-depsc');
close all;

figure;
plot(log(gamma),average_mse_train_10,'b-*');
hold on; grid on;
plot(log(gamma),average_mse_test_10,'r-o');
grid on;
xlabel('{log(\gamma)}','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE-train_{10}','MSE-test_{10}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex4c_mse_200_10','-depsc');
close all;

