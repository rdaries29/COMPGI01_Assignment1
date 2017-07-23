function [w_estimator, mse_train, mse_valid] = calculateLSRex6(x,w,n,train_len,gamma, k)

y = x*w + n;

x = x(1:train_len, :);
y = y(1:train_len, :);

[dim1, dim2] = size(x);

k_width = dim1/5;

% K-Fold Slicing and segmentation
x_valid = x((k-1)*k_width + 1:k*k_width, :);
x_train = cat( 1, x(1:(k-1)*k_width, :), x(k*k_width +1 :end,:));

y_valid = y((k-1)*k_width + 1:k*k_width, :);
y_train = cat( 1, y(1:(k-1)*k_width, :), y(k*k_width +1 :end,:));

% Compute weight vector
w_estimator = (x_train'*x_train + gamma*train_len*eye(size(w,1)))\(x_train'*y_train);

% Compute MSE for train and test sets
mse_train = meanSquareError(w_estimator,x_train,y_train,(train_len-k_width));
mse_valid = meanSquareError(w_estimator,x_valid,y_valid,k_width);


end