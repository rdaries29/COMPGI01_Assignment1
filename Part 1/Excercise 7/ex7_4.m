function [best_gamma_100,best_gamma_10,result_mse_test_100,result_mse_test_10] = ex7_4(x,w,n,gamma)

addpath('../library')

mse_train_100 = [];
mse_train_10 = [];
% Evaluate performance of test set
for i=1: size(gamma,2)
    [result_w_100_j, result_mse_train_100_j, result_mse_test_100_j] = calculateLSRex4(x,w,n,100,500,gamma(i));
    mse_train_100 = [mse_train_100, result_mse_train_100_j];
    
    [result_w_10_j, result_mse_train_10_j, result_mse_test_10_j] = calculateLSRex4(x,w,n,10,500,gamma(i));
    mse_train_10 = [mse_train_10, result_mse_train_10_j];
    
end
% Find the gamma that minimizes the MSE of training set
[min_train_error_100,min_idx_100] = min(mse_train_100);
best_gamma_100 = gamma(min_idx_100);

[min_train_error_10,min_idx_10] = min(mse_train_10);
best_gamma_10 = gamma(min_idx_10);

% Evaluate performance on test vector
[result_w_100, result_mse_train_100, result_mse_test_100] = calculateLSRex4(x,w,n,100,500,best_gamma_100);
[result_w_10, result_mse_train_10, result_mse_test_10] = calculateLSRex4(x,w,n,10,500,best_gamma_10);







