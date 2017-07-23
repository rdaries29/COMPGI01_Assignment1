%------------------------------------------------------------%
% Course : Supervised Learning
% Assignment : 1  
% Excercise 2
% Programmers: Russel Daries, Nitish Mutha
% Student ID: 16079408, 15113106
% Question: 2
% Section: Part 1
% Description: Least Squares Regression
% ------------------------------------------------------------------%

clear all
close all
clc

addpath('../library')

%% Ex2(a)
w = randn(10,1);
x = randn(600,10);
n = randn(600,1);

%% Ex2(b)
[part_b_result_w,part_b_result_mse_train,part_b_result_mse_test] = calculateLSRex2(x,w,n,100,500);
[part_c_result_w,part_c_result_mse_train,part_c_result_mse_test] = calculateLSRex2(x,w,n,10,500);

% Initilizae variables
result_100 = [];
result_10 = [];
w_100 = [];
w_10 = [];
mse_train_100 = [];
mse_train_10 = [];
mse_test_100 = [];
mse_test_10 =[];

for j = 1:200
    % Generate dataset
    w_j = randn(10,1);
    x_j = randn(600,10);
    n_j = randn(600,1);
    
    % Store resultant calculation
    % 100 training examples
    [result_w_100_j, result_mse_train_100_j, result_mse_test_100_j] = calculateLSRex2(x_j,w_j,n_j,100,500);
    w_100 = [w_100; result_w_100_j];
    mse_train_100 = [mse_train_100 ; result_mse_train_100_j];
    mse_test_100 = [mse_test_100; result_mse_test_100_j];
    
    % 10 training example
    [result_w_10_j, result_mse_train_10_j, result_mse_test_10_j] = calculateLSRex2(x_j,w_j,n_j,10,500);
    w_10 = [w_10; result_w_10_j];
    mse_train_10 = [mse_train_10 ; result_mse_train_10_j];
    mse_test_10 = [mse_test_10; result_mse_test_10_j];
    
end

% Average MSE for 10,100 training examples
average_mse_train_100 = mean(mse_train_100)
average_mse_test_100 = mean(mse_test_100)

% Average MSE on the test sets
average_mse_train_10 = mean(mse_train_10)
average_mse_test_10 = mean(mse_test_10)

% Code to plot resultant figures

figure;
h1 = plot(mse_train_10,'k*')
hold on
h2 = plot(mse_train_100,'b+')
hold on
h3 =plot(average_mse_train_10*ones(length(mse_train_10),1),'r--')
hold on
h4 = plot(average_mse_train_100*ones(length(mse_test_100),1),'m--')
xlabel('Trial number','FontSize',15)
ylabel('Mean Squared Error','FontSize',15)
grid on;
set(gcf, 'Color', 'w');
leg=legend('train10_{err}','train100_{err}','train-err10_{avg}','train-err100_{avg}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('ex2b_err_train','-depsc')
close all;


figure
plot(mse_test_10,'k*')
grid on;
hold on
plot(mse_test_100,'b+')
hold on
plot(average_mse_test_10*ones(length(mse_test_10),1),'m--')
hold on
plot(average_mse_test_100*ones(length(mse_test_100),1),'r--')
hold on
xlabel('Trial number','FontSize',15)
ylabel('Mean Squared Error','FontSize',15)
set(gcf, 'Color', 'w');
leg=legend('test10_{err}','test100_{avg}','test10-err_{avg}','test100-err_{avg}','Location','Best');
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
ylim([0.0 3e3])
print('ex2b_err_test','-depsc')
close all;

figure;
plot(w_10,'k*')
hold on
plot(w_100,'b+')
hold on
plot(mean(w_10)*ones(length(w_10),1),'r--')
hold on
plot(mean(w_100)*ones(length(w_100),1),'m--')
xlabel('Trial number','FontSize',15)
ylabel('Weight Estimate (w)','FontSize',15)
grid on;
set(gcf, 'Color', 'w');
leg=legend('w-train10','w-train100','w-train10_{avg}','w-train100_{avg}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('ex2b_wtrain','-depsc')
close all;




