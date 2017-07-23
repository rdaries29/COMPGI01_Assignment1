%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Stuart Daries, Nitish Mutha
% Student ID: 16079408 ,15113106
% Question: 7
% Section: Part 1
% Description: Comparing tuning-methods for regulrisation parameters
% ------------------------------------------------------------------%

%%
clear all
close all
clc

addpath('../library')

gamma_power = -6:1:3;
gamma = 10.^gamma_power;

%% Initialize variables of interest
test_4_100 = zeros(200,1);
test_4_10 = zeros(200,1);

test_5_100 = zeros(200,1);
test_5_10 = zeros(200,1);

test_6_100 = zeros(200,1);
test_6_10 = zeros(200,1);

all_gamma_4_100 = zeros(200,1);
all_gamma_4_10 = zeros(200,1);

all_gamma_5_100 = zeros(200,1);
all_gamma_5_10 = zeros(200,1);

all_gamma_6_100 = zeros(200,1);
all_gamma_6_10 = zeros(200,1);

%% Complete 200 independent trials
for j = 1:200
    
    w_j = randn(10,1);
    x_j = randn(600,10);
    n_j = randn(600,1);
    
    %% Select best gamma by minimizing training error
    [gamma_4_100,gamma_4_10,mse_test_4_100,mse_test_4_10] = ex7_4(x_j,w_j,n_j,gamma);
    %% Select best gammma by minimizing validation set error
    [gamma_5_100,gamma_5_10,mse_test_5_100,mse_test_5_10] = ex7_5(x_j,w_j,n_j,gamma);
    %% Select best gamma using 5-fold cross-validation
    [gamma_6_100,gamma_6_10,mse_test_6_100,mse_test_6_10] = ex7_6(x_j,w_j,n_j,gamma);
    
    % Assigning resultant mse test values to their respective outputs
    test_4_100(j) = mse_test_4_100;
    test_4_10(j) = mse_test_4_10;
    
    test_5_100(j) = mse_test_5_100;
    test_5_10(j) = mse_test_5_10;
    
    test_6_100(j) = mse_test_6_100;
    test_6_10(j) = mse_test_6_10;
    
    % Assignming optimal gamma values to vectors
    all_gamma_4_100(j) = gamma_4_100;
    all_gamma_4_10(j) = gamma_4_10;
    
    all_gamma_5_100(j) = gamma_5_100;
    all_gamma_5_10(j) = gamma_5_10;
    
    all_gamma_6_100(j) = gamma_6_100;
    all_gamma_6_10(j) = gamma_6_10;
end

%% Average MSE values
avg_test_4_100 = mean(test_4_100)
avg_test_4_10 = mean(test_4_10)

avg_test_5_100 = mean(test_5_100)
avg_test_5_10 = mean(test_5_10)

avg_test_6_100 = mean(test_6_100)
avg_test_6_10 = mean(test_6_10)

%% Average gamma values
avg_gamma_4_100 = mean(all_gamma_4_100)
avg_gamma_4_10 = mean(all_gamma_4_10)

avg_gamma_5_100 = mean(all_gamma_5_100)
avg_gamma_5_10 = mean(all_gamma_5_10)

avg_gamma_6_100 = mean(all_gamma_6_100)
avg_gamma_6_10 = mean(all_gamma_6_10)

%% Standard deviation calculation
sd_4_100 = standard_deviation(avg_test_4_100,test_4_100)
sd_4_10 = standard_deviation(avg_test_4_10,test_4_10)

sd_5_100 = standard_deviation(avg_test_5_100,test_5_100)
sd_5_10 = standard_deviation(avg_test_5_10,test_5_10)

sd_6_100 = standard_deviation(avg_test_6_100,test_6_100)
sd_6_10 = standard_deviation(avg_test_6_10,test_6_10)

%% plots
figure;
plot(test_4_100,'r*');
hold on;
plot(test_5_100,'b+')
hold on
plot(test_6_100,'go');
grid on;
set(gcf, 'Color', 'w');
xlabel('Trials','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{min-train}','MSE_{min-valid}','MSE_{min-cross}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex7_mse_compare_tuning_100','-depsc');
close all;

figure;
plot(test_4_10,'r*');
hold on;
plot(test_5_10,'b+')
hold on
plot(test_6_10,'go');
grid on;
set(gcf, 'Color', 'w');
xlabel('Trials','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{min-train}','MSE_{min-valid}','MSE_{min-cross}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex7_mse_compare_tuning_10','-depsc');
close all;

figure;
plot(all_gamma_4_10,'r*');
hold on;
plot(all_gamma_5_10,'b+')
hold on
plot(all_gamma_6_10,'go');
grid on;
set(gcf, 'Color', 'w');
xlabel('Trials','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{min-train}','MSE_{min-valid}','MSE_{min-cross}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex7_mse_compare_tuning_gamma_10','-depsc');
close all;

figure;
plot(all_gamma_4_100,'r*');
hold on;
plot(all_gamma_5_100,'b+')
hold on
plot(all_gamma_6_100,'go');
grid on;
set(gcf, 'Color', 'w');
xlabel('Trials','FontSize',15);
ylabel('Mean Square Error','FontSize',15);
leg=legend('MSE_{min-train}','MSE_{min-valid}','MSE_{min-cross}','Location','Best');
set(leg,'FontSize',15);
set(gca,'YMinorTick','on');
grid minor
axis tight;
print('ex7_mse_compare_tuning_gamma_100','-depsc');
close all;
