%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Daries, Nitish Mutha
% Student ID: 16079408, 15113106
% Question: 1
% Section: Part 1
% Description: Least Squares Regression
% ------------------------------------------------------------------%

% clearing memory
clear all
close all
clc

addpath('../library')

%% Ex1(a)

w = randn(1,1);
x = randn(600,1);
n = randn(600,1);

%% Ex1(b)
part_b_result = calculateLSR(x,w,n,100,500)


%% Ex1(c)
part_c_result = calculateLSR(x,w,n,10,500)

%% Ex1(d)

result_100 = [];
result_10 = [];
for j = 1:200
    w_j = randn(1,1);
    x_j = randn(600,1);
    n_j = randn(600,1);
    result_100 = [result_100 ; calculateLSR(x_j,w_j,n_j,100,500)];
    result_10 = [result_10 ; calculateLSR(x_j,w_j,n_j,10,500)];
end

average_result_100 = mean(result_100)
average_result_10 = mean(result_10)

% Code to plot resultant figures

figure;
plot(result_10(:,2),'k*')
hold on
plot(result_100(:,2),'b+')
hold on
plot(average_result_10(2)*ones(length(result_10),1),'r--')
hold on
plot(average_result_100(2)*ones(length(result_100),1),'m--')
xlabel('Trial number','FontSize',15)
ylabel('Mean Squared Error','FontSize',15)
grid on;
set(gcf, 'Color', 'w');
leg=legend('train10_{err}','train100_{err}','train-err10_{avg}','train-err100_{avg}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('ex1d_err_train','-depsc')
close all;


figure
plot(result_10(:,3),'k*')
grid on;
hold on
plot(result_100(:,3),'b+')
hold on
plot(average_result_10(2)*ones(length(result_10),1),'c--')
hold on
plot(average_result_100(3)*ones(length(result_100),1),'r--')
hold on
xlabel('Trial number','FontSize',15)
ylabel('Mean Squared Error','FontSize',15)
set(gcf, 'Color', 'w');
leg=legend('test10_{err}','test100_{avg}','test10-err_{avg}','test100-err_{avg}','Location','Best');
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
ylim([0.8 2.2])
print('ex1d_err_test','-depsc')
close all;

figure;
plot(result_10(:,1),'k*')
hold on
plot(result_100(:,1),'b+')
hold on
plot(average_result_10(1)*ones(length(result_10),1),'r--')
hold on
plot(average_result_100(1)*ones(length(result_100),1),'m--')
xlabel('Trial number','FontSize',15)
ylabel('Weight Estimate (w)','FontSize',15)
grid on;
set(gcf, 'Color', 'w');
leg=legend('w-train10','w-train100','w-train-err10_{avg}','w-train-err100_{avg}','Location','Best')
set(leg,'FontSize',15)
set(gca,'YMinorTick','on')
grid minor
axis tight;
print('ex1d_err_wtrain','-depsc')
close all;


