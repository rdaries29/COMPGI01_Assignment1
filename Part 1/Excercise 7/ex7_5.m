function [best_gamma_100,best_gamma_10,result_mse_test_100,result_mse_test_10] = ex7_5(x,w,n,gamma)

addpath('../library')

mse_valid_100 = [];
mse_valid_10 = [];
% Evaluate performance of training and validation sets of data
for i=1: size(gamma,2)
    [result_w_100_j, result_mse_train_100_j,result_mse_valid_100_j, result_mse_test_100_j] = calculateLSRex5(x,w,n,80,20,500,gamma(i));
    mse_valid_100 = [mse_valid_100, result_mse_valid_100_j];
    
    [result_w_10_j, result_mse_train_10_j,result_mse_valid_10_j, result_mse_test_10_j] = calculateLSRex5(x,w,n,8,2,500,gamma(i));
    mse_valid_10 = [mse_valid_10, result_mse_valid_10_j];
    
end
% Find the gamma that minimizes the MSE of validation set
[min_mse_100,idx_min_100] = min(mse_valid_100);
best_gamma_100 = gamma(idx_min_100);

[min_mse_10,idx_min_10] = min(mse_valid_10);
best_gamma_10 = gamma(idx_min_10);

% Evaluate performance on test vector
[w_test_100, result_mse_test_100] = calculateLSRTest(x,w,n,80,20,500,best_gamma_100);
[w_test_10, result_mse_test_10] = calculateLSRTest(x,w,n,8,2,500,best_gamma_10);



