%-------------------------------------------------------------------%
% Module: GI01 - Supervised Learning
% Assignment : Coursework 1
% Author : Russel Stuart Daries, Nitish Mutha
% Student ID: 16079408 ,15113106
% Question: 10
% Section: Part 1
% Description: Kernel Ridge Regression.
% ------------------------------------------------------------------%

close all;
clear;
clc;

addpath('../library')

load('boston.mat');

%% Ex10(c) and (d)

% Create gamma vector
gamma_exp = -40:-26;
gamma = 2.^gamma_exp;

% Create sigma vector
sigma_exp = 7:0.5:13;
sigma = 2.^sigma_exp;

% 20 trials
for j=1:20
    
    % Random sampling of data
    data_train = datasample(boston,round(2/3*size(boston,1)),1,'Replace',false);
    data_test = setdiff(boston, data_train, 'rows');
    
    % Training set
    x_train_j = data_train(:,1:13);
    y_train_j = data_train(:,14);
    
    %Test set
    x_test_j = data_test(:,1:13);
    y_test_j = data_test(:,14);
    
    %Dimension of training set x values
    [dim1, dim2] = size(x_train_j);
    
    k_mse = [];
    % Perform 5-fold cross validation
    for k=1:5
        
        k_width = round(dim1/5);
        
        % Segment x-train data for training
        x_train = cat( 1, x_train_j(1:(k-1)*k_width, :), x_train_j(k*k_width +1 :end,:));
        % Segment x-train data for validation
        x_valid = x_train_j((k-1)*k_width + 1:k*k_width, :);
        
        % Segment x-train data for training
        y_train = cat( 1, y_train_j(1:(k-1)*k_width, :), y_train_j(k*k_width +1 :end,:));
        % Segment x-train data for validation
        y_valid = y_train_j((k-1)*k_width + 1:k*k_width, :);
        
        [train_dim1, train_dim2] = size(x_train);
        
        K = zeros(dim1,dim1,size(sigma,2),1);
        % From Gram matrix
        for s=1:size(sigma,2)
            K(:,:,s) = calK(x_train,x_valid,sigma(s));
        end
        
        %Vary selection of gamma
        for g=1:size(gamma,2)
            %Vary selection of sigma
            for ss=1:size(sigma,2)
                current_k = K(:,:,ss);
                alpha = kridgereg(current_k(1:train_dim1,1:train_dim1),y_train,gamma(g));
                %Calculate resultant mse for each fold,each gamma and each
                %sigma selection and store it into 3D variable
                k_mse(g,ss,k) = dualcost(current_k(train_dim1+1:end,1:train_dim1) ,y_valid,alpha);
            end
        end
    end
    % Calculating the mean MSE over the 5 folds of data
    mean_k_mse = mean(k_mse,3);
    
    %% Ex10 (c)
    %plot the cross validation error (once as asked in 10_c)
    if(j == 1)
        surf(sigma',gamma',mean_k_mse)
        colorbar;
        set(gcf, 'Color', 'w');
        xlabel('\sigma','FontSize',15);
        ylabel('\gamma','FontSize',15);
        zlabel('MSE_{cross-validaion}','FontSize',15);
        set(gca,'YMinorTick','on');
        grid minor
        axis tight;
        print('ex10_cv_mse','-depsc');
        close all;
    end
    
    % Find the combination of gamma and sigma that produces smalled mse
    [min_gamma_indx, min_sigma_indx] = find(mean_k_mse == min(min(mean_k_mse)));
    
    % Using optimal gamma and sigma values to calculate optimal alpha
    % values
    K_optimal = calK(x_train_j,x_test_j,sigma(min_sigma_indx));
    alpha_optimal = kridgereg(K_optimal(1:dim1,1:dim1),y_train_j,gamma(min_gamma_indx));
    
    %training error
    mse_training(j) = dualcost(K_optimal(1:dim1,1:dim1) ,y_train_j,alpha_optimal);
    
    %testing error (Using K_test from lower left hand corner of gram matrix)
    mse_testing(j) = dualcost(K_optimal(dim1+1:end,1:dim1) ,y_test_j,alpha_optimal);
    
end
% Calculate the mean and standard deviation of resultant mse of training
% and test set
avg_mse_training = mean(mse_training)
avg_mse_testing = mean(mse_testing)

std_mse_training = std(mse_training)
std_mse_testing = std(mse_testing)