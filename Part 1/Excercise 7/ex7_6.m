function [best_gamma_100,best_gamma_10,result_mse_test_100,result_mse_test_10] = ex7_6(x,w,n,gamma)

addpath('../library')

result_mse_valid_100 = [];
result_mse_valid_10 = [];

% Perform 5-fold cross-validation
for j = 1:5
    
    mse_valid_100 = [];
    mse_valid_10 = [];
    
    for i=1: size(gamma,2)
        [result_w_100_j, result_mse_train_100_j,result_mse_valid_100_j] = calculateLSRex6(x,w,n,100,gamma(i),j);
        mse_valid_100 = [mse_valid_100, result_mse_valid_100_j];
        
        [result_w_10_j, result_mse_train_10_j,result_mse_valid_10_j] = calculateLSRex6(x,w,n,10,gamma(i),j);
        mse_valid_10 = [mse_valid_10, result_mse_valid_10_j]; 
    end
    
    result_mse_valid_100 = [result_mse_valid_100 ; mse_valid_100];
    result_mse_valid_10 = [result_mse_valid_10 ; mse_valid_10];
end

% Find the gamma that minimizes the mean cross-validation MSE
[min_mse_100,min_idx_100] = min(mean(result_mse_valid_100));
best_gamma_100 = gamma(min_idx_100);

[min_mse_10,min_idx_10] = min(mean(result_mse_valid_10));
best_gamma_10 = gamma(min_idx_10);

% Evaluate performance on test vector
[result_w_100, result_mse_train_100,result_mse_test_100] = calculateLSRex4(x,w,n,100,500,best_gamma_100);
[result_w_10, result_mse_train_10,result_mse_test_10] = calculateLSRex4(x,w,n,10,500,best_gamma_10);








